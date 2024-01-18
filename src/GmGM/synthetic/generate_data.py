from __future__ import annotations

import numpy as np
from scipy.stats import invwishart, wishart
from scipy.stats import multivariate_normal, ortho_group
from scipy.linalg import toeplitz

from typing import Optional, Callable, Literal
from ..typing import Modality, Axis, MaybeDict
from ..dataset import Dataset


"""
============================================================
==================== Utility Functions =====================
============================================================
"""

def kron_sum(A, B):
    """
    Computes the kronecker sum of two square input matrices
    
    Note: `scipy.sparse.kronsum` is a thing that would
    be useful - but it seems that `scipy.sparse` is not
    yet a mature library to use.
    """
    a, _ = A.shape
    b, _ = B.shape
    return np.kron(A, np.eye(b)) + np.kron(np.eye(a), B)

def kron_prod(A, B):
    """
    Computes the kronecker product.  There is a built in
    np.kron, but it's slow.  Can use a broadcasting trick to
    speed things up.
    
    Trick from greggo's answer in:
    https://stackoverflow.com/questions/7193870/speeding-up-numpy-kronecker-products
    """
    a1, a2 = A.shape
    b1, b2 = B.shape
    return (
        A[:, np.newaxis, :, np.newaxis]
        * B[np.newaxis, :, np.newaxis, :]
    ).reshape((a1*b1, a2*b2))

def kron_prod_diag(
    *Λs: list[np.ndarray]
) -> np.ndarray:
    if len(Λs) == 1:
        return Λs[0]
    elif len(Λs) == 2:
        return np.kron(Λs[0], Λs[1])
    else:
        return np.kron(Λs[0], kron_prod_diag(Λs[1:]))

def kron_sum_diag_fast(
    *lams: list[np.ndarray]
) -> np.ndarray:
    # Setup
    ds = [len(lam) for lam in lams]
    d_lefts = np.cumprod([1] + ds[:-1]).astype(int)
    d_rights = np.cumprod([1] + ds[::-1])[-2::-1].astype(int)
    total = d_rights[0] * ds[0]
    out = np.zeros(total)
    
    
    for ell, lam in enumerate(lams):
        add_like_kron_sum(
            out,
            lam,
            ds[ell],
            d_lefts[ell],
            d_rights[ell]
        )
        
    return out

def kron_sum_squared_diag_fast(
    *Λs: list[np.ndarray]
) -> np.ndarray:
    return kron_sum_diag_fast(*Λs) ** 2

def add_like_kron_sum(
    cur_kron_sum: np.ndarray,
    to_add: np.ndarray,
    d: int,
    d_left: int,
    d_right: int
) -> None:
    """
    !!!!Modifies cur_kron_sum in place!!!!
    
    Let X[+]Y be the Kronecker sum
    of diagonal matrices.
    Sometimes we want to find X[+](Y+Z)
    for diagonal Z
    
    This is a way to update our pre-computed
    X[+]Y to incorporate the additive Z.

    Old typing hint:
        cur_kron_sum: "Kronsummed matrix",
        to_add: "What to add to matrix",
        ell: "Dimension to add along",
        d, d_left, d_right
    """
    # We're gonna be really naughty here and use stride_tricks
    # This is going to reshape our vector in a way so that the elements
    # we want to affect are batched by the first two dimensions
    sz = to_add.strides[0]
    toset = np.lib.stride_tricks.as_strided(
        cur_kron_sum,
        shape=(
            d_left, # The skips
            d_right, # The blocks
            d # What we want
        ),
        strides=(
            sz * d * d_right,
            sz * 1,
            sz * d_right,
        )
    )
    toset += to_add

def threshold_matrix(
    Psi: np.ndarray,
    sparsity: float,
) -> np.ndarray:
    Psabs = np.abs(Psi)
    diagonal = np.diag(Psi).copy()
    np.fill_diagonal(Psabs, 0)
    quant = np.quantile(Psabs, 1-sparsity)
    Psi[Psabs < quant] = 0
    np.fill_diagonal(Psi, diagonal)
    return Psi

def threshold_dictionary(
    Psi: dict[Axis, np.ndarray],
    sparsity: dict[Axis, float],
) -> np.ndarray:
    return {
        axis: threshold_matrix(Psi[axis], sparsity[axis])
        for axis in Psi
    }

def make_sparse_small_spectrum(
    Psi: np.ndarray,
    sparsity: float,
    spectrum_size: int = 20
) -> np.ndarray:
    diagonal = np.diag(Psi).copy()
    U, S, Vh = np.linalg.svd(Psi)
    U = U[:, :spectrum_size]
    S = S[:spectrum_size]
    Vh = Vh[:spectrum_size, :]
    Psi = U @ np.diag(S) @ Vh
    np.fill_diagonal(Psi, diagonal)
    return threshold_matrix(Psi, sparsity)

"""
============================================================
==================== GENERATE DATA =========================
============================================================
"""

def fast_kronecker_normal(
    Psis: list[np.ndarray],
    size: int,
    axis_join: Callable[[list[np.ndarray]], np.ndarray]
        | Literal["Kronecker Sum", "Kronecker Product", "Kronecker Sum Squared"],
    fail_if_not_posdef: bool = False,
    mean: np.array = None,
) -> np.ndarray:
    """
    Inputs:
        Psis: List of (d_i, d_i) precision matrices, of length K >= 2
        size: Number of samples
        axis_join:
            method of combining eigenvalues
            can be anything, but here are values for common distributions
            * Kronecker Product Normal Distribution:
                kron_prod_diag
            * Kronecker Sum Normal Distribution:
                kron_sum_diag_fast
            * Kronecker Sum Squared Normal Distribution:
                kron_sum_squared_diag_fast
        fail_if_not_posdef:
            If True, raise Exception if any of the Psis is not positive definite
        mean: Mean of the distribution

    Outputs:
        Xs: Sample of Kronecker sum structured normal distribution
    """

    if isinstance(axis_join, str):
        if axis_join == "Kronecker Sum":
            axis_join = kron_sum_diag_fast
        elif axis_join == "Kronecker Product":
            axis_join = kron_prod_diag
        elif axis_join == "Kronecker Sum Squared":
            axis_join = kron_sum_squared_diag_fast
        else:
            raise ValueError(f"Unknown method {axis_join}")

    K = len(Psis)
    ds = [Psi.shape[0] for Psi in Psis]
    vs, Vs = zip(*[np.linalg.eigh(Psi) for Psi in Psis])
    diag_precisions = axis_join(*vs)
    
    # Check if positive definite
    min_diag = diag_precisions.min()
    if min_diag < 0:
        if fail_if_not_posdef:
            raise Exception("KS of Psis not Positive Definite")
        diag_precisions -= (min_diag-1)
    
    # Sample from diagonalized, vectorized distribution
    z = multivariate_normal(cov=1).rvs(
        size=size*np.prod(ds)
    ).reshape(size, np.prod(ds)) / np.sqrt(diag_precisions)
    
    # Reshape into a tensor
    Xs: np.ndarray = z.reshape(size, *ds)
    
    # Undiagonalize the distribution
    for k in range(K):
        Xs = np.moveaxis(
            np.moveaxis(Xs, k+1, -1) @ Vs[k].T,
            -1,
            k+1
        )

    if mean is not None:
        Xs += mean.reshape(1, *ds)

    return Xs



class PrecMatMask:
    """
    The sparsity pattern for a precision matrix
    """

    def __init__(self):
        pass

    def generate(
        self,
        n: int
    ) -> np.ndarray:
        """
        Inputs:
            n: Number of rows/columns of output
        
        Outputs:
            (n, n) binary positive definite mask matrix
        """
        raise NotImplementedError
    
    def _neg_laplace(self, mat: np.ndarray) -> np.ndarray:
        """
        Computes the negative laplacian of a matrix
        """
        mat = mat.copy()
        np.fill_diagonal(mat, 0)
        return np.diag(mat.sum(axis=1)) - mat

class PrecMatOnes(PrecMatMask):
    """
    Matrix full of ones
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def generate(
        self,
        n: int,
        n_comps: Optional[int] = None
    ) -> np.ndarray:
        if n_comps is not None:
            raise ValueError("n_comps not supported for PrecMatOnes")
        return np.ones((n, n))
    
    def __repr__(self) -> str:
        return "PrecMatOnes()"
    
class PrecMatIndependent(PrecMatMask):
    """
    Identity matrix
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def generate(
        self,
        n: int,
        n_comps: Optional[int] = None
    ) -> np.ndarray:
        _n = n_comps if n_comps is not None else n
        to_return = np.zeros((n, n))
        to_return[:_n, :_n] = np.eye(_n)
        return to_return
    
    def __repr__(self) -> str:
        return "PrecMatIndependent()"
    
class PrecMatErdosRenyiGilbert(PrecMatMask):
    """
    Eros-Renyi-Gilbert graph,
    i.e. each edge is bernoulli identical and independant
    """
    
    def __init__(
        self,
        *,
        edge_probability: float = 0.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.edge_probability = edge_probability

    def generate(
        self,
        n: int,
        n_comps: Optional[int] = None
    ) -> np.ndarray:
        
        #to_return = 1e-6 * np.eye(n)
        to_return = np.zeros((n, n))
        n_comps = n_comps if n_comps is not None else n
        unsymmetrized = (np.random.rand(n_comps, n_comps) < self.edge_probability).astype(int)

        # Symmetrize!
        unsymmetrized[np.tril_indices(n_comps)] = 0
        symmetrized = unsymmetrized + unsymmetrized.T
        symmetrized = symmetrized.astype(float)

        #to_return = self._neg_laplace(symmetrized)

        # if n_comps is not None:
        #     V = ortho_group(n).rvs()[:_n, :]
        #     to_return = V.T @ to_return @ V
        #     to_return = threshold_matrix(np.abs(to_return), self.edge_probability)

        # if n_comps is not None:
        #     to_fill = np.diag(to_return).copy()
        #     to_fill[:n_comps] = to_fill[:n_comps]**4
        #     np.fill_diagonal(to_return, to_fill)

        to_return[:n_comps, :n_comps] = self._neg_laplace(symmetrized)

        return to_return
    
    def __repr__(self) -> str:
        return f"PrecMatErdosRenyiGilbert(edge_probability={self.edge_probability})"
    
class PrecMatAutoregressive(PrecMatMask):
    """
    Autoregressive matrix
    """
    
    def __init__(
        self,
        *,
        p: int,
        **kwargs
    ):
        """
        Inputs:
            p: An AR(p) matrix
        """
        super().__init__(**kwargs)
        self.p = p
    
    def generate(
        self,
        n: int,
        n_comps: Optional[int] = None
    ) -> np.ndarray:
        """
        Creates a Toeplitz matrix with p+1 nonzero off-diagonals
        (where the +1 is because diagonal should be nonzero)
        """
        if n_comps is not None:
            raise ValueError("n_comps not supported for PrecMatAutoregressive")
        diags = np.zeros(n)
        diags[:self.p+1] = 1
        return self._neg_laplace(toeplitz(diags))
    
    def __repr__(self) -> str:
        return f"PrecMatAutoregressive(p={self.p})"
    
class PrecMatBlob(PrecMatMask):
    """
    Matrix with a blob of fully connectedness
    """
    
    def __init__(
        self,
        *,
        edge_probability: float,
        **kwargs
    ):
        """
        Inputs:
            probability: Probability that an edge is connected to the blob
        """
        super().__init__(**kwargs)
        self.edge_probability = edge_probability
    
    def generate(
        self,
        n: int,
        n_comps: Optional[int] = None
    ) -> np.ndarray:
        """
        Creates a matrix with a cluster such that:
            all vertices in the cluster are connected to each other
            all vertices outside the cluster are connected to nothing
        """
        if n_comps is not None:
            raise ValueError("n_comps not supported for PrecMatBlob")
        b: np.ndarray = np.random.rand(n).reshape(n, 1) < np.sqrt(self.edge_probability)
        cov_mat: np.ndarray = (b @ b.T).astype(float)
        np.fill_diagonal(cov_mat, 0)
        cov_mat += np.eye(n)
        return self._neg_laplace(cov_mat)
    
    def __repr__(self) -> str:
        return f"PrecMatBlob(probability={self.edge_probability})"

class PrecMatGenerator:
    """
    My new-style precision matrix generator
    """

    def __init__(
        self,
        *,
        core_type: Literal[
            "invwishart",
            "wishart",
            "coreless"
        ] = "coreless",
        mask: PrecMatMask | Literal[
            "Erdos-Renyi-Gilbert",
            "Autoregressive",
            "Ones",
            "IID",
            "Blob"
        ] = "Erdos-Renyi-Gilbert",
        n_comps: Optional[int] = None,
        scale: float = 1
    ):
        """
        There are two components to the generator:
            1. The core distribution Psi
            2. The mask distribution M

        We then return Psi ∘ M, where ∘ is the Hadamard product.
        By the Schur Product Theorem, this is guaranteed to be posdef
            if Psi and M are posdef.

        The core controls the magnitudes of the distribution.
        
        The mask controls the sparsity of the distribution, so we can have
            it be a binary matrix.  This would not be posdef unless we scale the
            diagonal M_{ii} to be any value larger than the sum of the ith row,
            but that is easy to do. (This is b/c then the mask is
            Diagonally Dominant and hence psodef).

            We are happy to mess with the diagonal, as it does not affect the
            graph structure.

        Inputs:
            Core:
                core_type:
                    "invwishart":
                        The core follows an inverse wishart distribution
                    "wishart":
                        The core follows a wishart distribution
                    "coreless":
                        The core is a matrix of ones, i.e. we just use the mask.
            Mask:
                mask_type:
                    "Eros-Renyi-Gilbert":
                        The mask follows an Erdos-Renyi-Gilbert distribution
                        i.e. every edge is i.i.d. Bernoulli(p)
                    "Autoregressive":
                        The mask follows an AR(p) distribution, i.e. each edge
                        is connected to the p edges before it
                    "Ones":
                        The mask is a matrix of ones, i.e. we just use the core
                    "Independent":
                        The mask is the identity matrix, i.e. nothing is connected
                    "Blob":
                        The mask contains a blob of fully connectedness
            n_comps:
                If not None, the result will be modified to have all but `n_comps` eigenvalues near zero.
            scale: multiply whole precision matrix by this value
        """

        if isinstance(mask, str):
            if mask == "Erdos-Renyi-Gilbert":
                mask = PrecMatErdosRenyiGilbert(edge_probability=0.5)
            elif mask == "Autoregressive":
                mask = PrecMatAutoregressive(2)
            elif mask == "Ones":
                mask = PrecMatOnes()
            elif mask == "IID":
                mask = PrecMatIndependent()
            elif mask == "Blob":
                mask = PrecMatBlob(probability=0.5)
            else:
                raise ValueError(f"Unknown mask type {mask}")
            
        self.core_type = core_type
        self.mask = mask
        self.scale = scale
        self.n_comps = n_comps

    def generate(
        self,
        n: int,
        readonly: bool = False
    ) -> np.ndarray:
        """
        Inputs:
            n: Number of rows/columns of output
        
        Outputs:
            (n, n) binary positive definite mask matrix
        """

        if self.core_type == "invwishart":
            core = invwishart.rvs(n, np.eye(n))
        elif self.core_type == "wishart":
            core = wishart.rvs(n, np.eye(n))
        elif self.core_type == "coreless":
            core = np.ones((n, n))
        else:
            raise ValueError(f"Unknown core type {self.core_type}")

        mask = self.mask.generate(n, n_comps=self.n_comps)

        output = self.scale * core * mask

        # if self.n_comps is not None:
        #     sparsity = np.count_nonzero(output) / output.size
        #     # A bit of a hack, but repeating this process seems to work
        #     for i in range(100):
        #         output = make_sparse_small_spectrum(output, sparsity, self.n_comps)

        if readonly:
            output.flags.writeable = False

        return output
    
    def __repr__(self) -> str:
        output = f"<PrecMatGenerator, core={self.core_type}, mask={self.mask}"
        if self.n_comps is not None:
            output += f", n_comps={self.n_comps}"
        output += ">"
        return output

class GroupedPrecMatGenerator(PrecMatGenerator):
    """
    Generates precision matrices with a block-diagonal structure
    with some baseline randomness added on
    """

    def __init__(
        self,
        groups: list[tuple[PrecMatGenerator, int]],
        *,
        baseline_core_type: str = "invwishart",
        baseline_mask: PrecMatMask | str = "IID",
        baseline_strength: float = 0.1
    ):
        super().__init__(
            core_type=baseline_core_type,
            mask=baseline_mask
        )
        self.baseline_strength = baseline_strength
        self.groups = groups

    def generate(self, n: int) -> np.ndarray:
        # Baseline noise
        baseline = self.baseline_strength * super().generate(n)

        # Keeps track of where to add into the baseline
        to_return = baseline
        prev = 0

        # Add groups on top of baseline
        for generator, size in self.groups:
            to_return[prev:prev+size, prev:prev+size] += generator.generate(size)
            prev += size

        return to_return
    
    def __repr__(self) -> str:
        rep_str = "<GroupedPrecMatGenerator, groups=\n"
        for group in self.groups:
            rep_str += f"\t{group}\n"
        return rep_str + ">"
    
class BaseDistribution:
    """
    Given a precision matrix, what distribution do we want our data to follow?
    Typically is Normal, but could be others
    """
    def __init__(self):
        pass

    # In Python 3.12, can add @override decorator here
    def generate(
        self,
        Psis: list[np.ndarray],
        num_samples: int,
        axis_join: Callable[[list[np.ndarray]], np.ndarray] |
            Literal["Kronecker Sum", "Kronecker Product", "Kronecker Sum Squared"],
    ) -> np.ndarray:
        raise NotImplementedError

class NormalDistribution(BaseDistribution):
    """
    Normal distribution
    """
    def __init__(self):
        super().__init__()

    def generate(
        self,
        Psis: list[np.ndarray],
        num_samples: int,
        axis_join: Callable[[list[np.ndarray]], np.ndarray] |
            Literal["Kronecker Sum", "Kronecker Product", "Kronecker Sum Squared"],
    ) -> np.ndarray:
        return fast_kronecker_normal(
            Psis,
            num_samples,
            axis_join=axis_join,
        )
    
    def __repr__(self) -> str:
        return "<Normal Distribution>"

class LogNormalDistribution(NormalDistribution):
    """
    Log normal distribution
    """
    def __init__(self, cutoff: Optional[float] = 100):
        """
        `cutoff` is the value at which we truncate the original normal distribution
        to prevent overflow errors
        """
        super().__init__()
        self.cutoff = cutoff

    def generate(
        self,
        Psis: list[np.ndarray],
        num_samples: int,
        axis_join: Callable[[list[np.ndarray]], np.ndarray] |
            Literal["Kronecker Sum", "Kronecker Product", "Kronecker Sum Squared"],
    ) -> np.ndarray:
        to_return = super().generate(
            Psis,
            num_samples,
            axis_join=axis_join,
        )
        to_return[to_return > self.cutoff] = self.cutoff
        return np.exp(to_return)
    
    def __repr__(self) -> str:
        return "<Log Normal Distribution>"

class ZiLNDistribution(LogNormalDistribution):
    """
    Zero-inflated log-normal distribution
    """
    def __init__(self, truncation):
        super().__init__()
        self.truncation = truncation

    def generate(
        self,
        Psis: list[np.ndarray],
        num_samples: int,
        axis_join: Callable[[list[np.ndarray]], np.ndarray] |
            Literal["Kronecker Sum", "Kronecker Product", "Kronecker Sum Squared"],
    ) -> np.ndarray:
        Ys = super().generate(
            Psis,
            num_samples,
            axis_join=axis_join,
        )
        Ys[Ys < self.truncation] = 0
        return Ys
    
    def __repr__(self) -> str:
        return f"<ZiLN Distribution, truncation={self.truncation}>"
    
class ZiLNMultinomial(ZiLNDistribution):
    """
    As in the paper:
    A zero inflated log-normal model for inference of sparse microbial association networks
    by Prost, Gazut, and Bruls
    """

    def __init__(self, truncation, num_reads):
        super().__init__(truncation)
        self.num_reads = num_reads

    def generate(
        self,
        Psis: list[np.ndarray],
        num_samples: int,
        axis_join: Callable[[list[np.ndarray]], np.ndarray] |
            Literal["Kronecker Sum", "Kronecker Product", "Kronecker Sum Squared"],
    ) -> np.ndarray:
        Ys = super().generate(
            Psis,
            num_samples,
            axis_join=axis_join,
        )
        for i, sample in enumerate(Ys):
            for j, row in enumerate(sample):
                # row represents the probability parameter of the multinomial distribution
                # n represents the total size of the multinomial distribution
                n = self.num_reads[j]
                Ys[i, j] = np.random.multinomial(n, row / row.sum())

        return Ys
    
    def __repr__(self) -> str:
        return f"<ZiLN Multinomial, truncation={self.truncation}, num_reads={self.num_reads}>"

class DatasetGenerator:
    """
    Generates potentially multimodal datasets
    """

    def __init__(
        self,
        *,
        structure: dict[Modality, tuple[Axis]],
        Psis: Optional[dict[Axis, np.ndarray]] = None,
        generator: Optional[dict[Axis, PrecMatGenerator]] = None,
        size: Optional[dict[Axis, int]] = None,
        batch_name: str = "",
        axis_join: MaybeDict[
            Modality,
            Callable[[list[np.ndarray]], np.ndarray] |
                Literal["Kronecker Sum", "Kronecker Product", "Kronecker Sum Squared"]
        ] = "Kronecker Sum",
        distribution: MaybeDict[
            Modality,
            Literal["Normal", "Log Normal"]
                | NormalDistribution
                | LogNormalDistribution
                | ZiLNDistribution
        ] = "Normal"
    ):
        self.axis_join = axis_join
        self.distribution = distribution
        self.batch_name = batch_name
        self.structure: dict[Modality, tuple[Axis]] = {
            name: (self.batch_name, *axes)
            for name, axes in structure.items()
        }
        self.axes: set[Axis] = {
            axis
            for axes in structure.values()
            for axis in axes
        }
        self.Psis = Psis
        self.generator = generator
        self.size = size


        if self.generator is None and self.Psis is None:
            raise ValueError("Must provide either Psis or generator")
        
        if self.size is None and self.Psis is None:
            raise ValueError("Must provide either Psis or size")
        
        if self.size is None:
            self.size = {
                axis: Psi.shape[0]
                for axis, Psi in self.Psis.items()
            }

        if self.Psis is None:
            self.Psis = {
                axis: generator.generate(self.size[axis])
                for axis, generator in self.generator.items()
            }

        # Convert distribution into a dict indexed by modality
        if isinstance(self.distribution, str) | isinstance(self.distribution, BaseDistribution):
            self.distribution = {
                name: self.distribution
                for name in self.structure.keys()
            }
        # Convert string-encoded distributions into the corresponding classes
        for key, value in self.distribution.items():
            if isinstance(value, str):
                if value == "Normal":
                    self.distribution[key] = NormalDistribution()
                elif value == "Log Normal":
                    self.distribution[key] = LogNormalDistribution()
                else:
                    raise ValueError(f"Unknown distribution {value}")
        
    def reroll_Psis(self, readonly: bool = False):
        if self.generator is None:
            raise ValueError("Cannot reroll Psis if generator is not provided")
        
        self.Psis = {
            axis: generator.generate(self.size[axis], readonly=readonly)
            for axis, generator in self.generator.items()
        }

    def generate(
        self,
        num_samples: MaybeDict[str, int] = 1,
        readonly: bool = False
    ) -> Dataset:
        """
        Inputs:
            num_samples: Dict of number of samples for each modality
            structure:
                Dict of tuples of axis names, keyed by modality
                --
                Each tuple is a shape of a tensor
                --
            ds:
                Dictionary of axis sizes
            sparsities:
                Dictionary of sparsities for each axis
            gen_type:
                "bernoulli" or "invwishart"
                --
                "bernoulli" generates a bernoulli distribution
                "invwishart" generates a (sparsified) inverse wishart distribution
                --
            mean:
                Mean of the distribution

        Outputs:
            Psis:
                Dictionary of precision matrices
            Ys:
                Dictionary of samples of Kronecker sum structured normal distribution


        Suppose we have a multiomics dataset of:
        300 people x 50 gut microbes x 12 timestamps
        300 people x 200 metabolites
        
        Then structure would be:
        {
            "microbiome": ("people", "microbes", "time"),
            "metabolome": ("people", "metabolites")
        }
        
        and ds would be:
        {
            "people": 300,
            "microbes": 50,
            "time": 12,
            "metabolites": 200
        }
        """
        
            
        Ys: dict = {name : None for name in self.structure.keys()}

        # If passed integer, make all datasets have the same number of samples,
        # with that integer being the number of samples.
        if isinstance(num_samples, int):
            num_samples = {name: num_samples for name in self.structure.keys()}

        for name, axes in self.structure.items():
            Ys[name] = self.distribution[name].generate(
                [self.Psis[axis] for axis in axes if axis is not self.batch_name],
                num_samples[name],
                axis_join=self.axis_join
            )
            if readonly:
                Ys[name].flags.writeable = False
            
        return Dataset(
            dataset=Ys,
            structure=self.structure,
            batch_axes={self.batch_name},
        )
    
    def __repr__(self) -> str:
        rep_str = "<DatasetGenerator, structure=\n"
        for name, axes in self.structure.items():
            rep_str += f"\t{name}: {axes}\n"
        rep_str += "size=\n"
        for name, size in self.size.items():
            rep_str += f"\t{name}: {size}\n"
        if self.generator is not None:
            rep_str += "generator=\n"
            for name, generator in self.generator.items():
                rep_str += f"\t{name}: {generator}\n"
        rep_str += "axis_join=\n"
        rep_str += f"\t{self.axis_join}\n"
        rep_str += "distribution=\n"
        for name, distribution in self.distribution.items():
            rep_str += f"\t{name}: {distribution}\n"
        return rep_str + ">"


"""
elif dist == "Log Normal" or isinstance(dist, LogNormalDistribution):
                Ys[name] = np.exp(fast_kronecker_normal(
                    [self.Psis[axis] for axis in axes if axis is not self.batch_name],
                    m[name],
                    axis_join=self.axis_join,
                ))
            elif isinstance(dist, ZiLNDistribution):
                Ys[name] = np.exp(fast_kronecker_normal(
                    [self.Psis[axis] for axis in axes if axis is not self.batch_name],
                    m[name],
                    axis_join=self.axis_join,
                ))
                Ys[name][Ys[name] < dist.truncation] = 0
            else:
                raise ValueError(f"Unknown distribution {self.distribution[name]}")
                """