from __future__ import annotations

import numpy as np
from scipy.stats import invwishart, wishart
from scipy.stats import multivariate_normal
from scipy.linalg import toeplitz

from typing import Optional, Callable, Literal
from .typing import Modality, Axis

# TODO: Move over to generating Dataset objects


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
            ell,
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
    cur_kron_sum: "Kronsummed matrix",
    to_add: "What to add to matrix",
    ell: "Dimension to add along",
    d, d_left, d_right
) -> None:
    """
    !!!!Modifies cur_kron_sum in place!!!!
    
    Let X[+]Y be the Kronecker sum
    of diagonal matrices.
    Sometimes we want to find X[+](Y+Z)
    for diagonal Z
    
    This is a way to update our pre-computed
    X[+]Y to incorporate the additive Z.
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

"""
============================================================
==================== GENERATE DATA =========================
============================================================
"""

def fast_ks_normal(
    Psis: list[np.ndarray],
    size: int,
    fail_if_not_posdef: bool = False,
    mean: np.array = None
) -> np.ndarray:
    """
    Inputs:
        Psis: List of (d_i, d_i) precision matrices, of length K >= 2
        size: Number of samples
        fail_if_not_posdef:
            If True, raise Exception if any of the Psis is not positive definite
        mean: Mean of the distribution

    Outputs:
        Xs: Sample of Kronecker sum structured normal distribution

    SUPERSEDED BY fast_kronecker_normal
    """
    K = len(Psis)
    ds = [Psi.shape[0] for Psi in Psis]
    vs, Vs = zip(*[np.linalg.eigh(Psi) for Psi in Psis])
    diag_precisions = kron_sum_diag_fast(*vs)
    
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

def fast_kronecker_normal(
    Psis: list[np.ndarray],
    size: int,
    method: Callable[[list[np.ndarray]], np.ndarray],
    fail_if_not_posdef: bool = False,
    mean: np.array = None,
) -> np.ndarray:
    """
    Inputs:
        Psis: List of (d_i, d_i) precision matrices, of length K >= 2
        size: Number of samples
        method:
            method of combining eigenvalues
            can be anything, but here are values for common distributions
            * Kronecker Product Normal Distribution:
                kron_prod_diag
            * Kronecker Sum Normal Distribution:
                kron_sum_diag_fast
            * Kronecker Squared Sum Normal Distribution:
                kron_sum_squared_diag_fast
        fail_if_not_posdef:
            If True, raise Exception if any of the Psis is not positive definite
        mean: Mean of the distribution

    Outputs:
        Xs: Sample of Kronecker sum structured normal distribution
    """
    K = len(Psis)
    ds = [Psi.shape[0] for Psi in Psis]
    vs, Vs = zip(*[np.linalg.eigh(Psi) for Psi in Psis])
    diag_precisions = method(*vs)
    
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

def generate_Psis(
    ds: dict[str, tuple[int]],
    *,
    sparsities: dict[str, float],
    gen_type: Literal["bernoulli", "invwishart"] = "bernoulli"
) -> dict[str, np.ndarray]:
    """
    Inputs:
        ds: Dictionary of dimensions of each mode
        sparsities:
            Dictionary of percent of nonzero edges in ground truth
            for each mode
        gen_type:
            "bernoulli" or "invwishart"
            --
            "bernoulli" generates a bernoulli distribution
            "invwishart" generates a (sparsified) inverse wishart distribution
            --
    
    Outputs:
        Dictionary of precision matrices
    """
    
    Psis: dict[str, np.ndarray] = {}
        
    for axis in ds.keys():
        Psi = generate_Psi(ds[axis], sparsities[axis], gen_type=gen_type)
        Psis[axis] = Psi
        
    return Psis

def generate_Psi(
    d: int,
    s: float,
    *,
    gen_type: "bernoulli or invwishart" = "bernoulli"
) -> np.ndarray:
    """
    Inputs:
        d: Dimension of current mode
        s: Sparsity of current mode
        gen_type:
            "bernoulli" or "invwishart"
            --
            "bernoulli" generates a bernoulli distribution
            "invwishart" generates a (sparsified) inverse wishart distribution
            --

    Outputs:
        * Precision matrix for current mode
    """
    
    if gen_type == "bernoulli":
        Psi: "(d, d)" = np.triu(bernoulli.rvs(p=s, size=(d, d)))
        Psi = Psi.T + Psi
        np.fill_diagonal(Psi, 1)
    elif gen_type == "invwishart":
        Psi = generate_sparse_invwishart_matrix(
            d,
            s*d**2 / 2,
            off_diagonal_scale=0.9,
            size=1,
            df_scale=1
        ).squeeze()
    else:
        raise Exception(f"Invalid input '{gen_type}'!")
    return Psi

def generate_synthetic_dataset(
    m: dict[str, int],
    structure: dict[str, tuple[str]],
    ds: dict[str, int],
    *,
    sparsities: dict[str, float],
    gen_type: "bernoulli or invwishart" = "bernoulli",
    mean: Optional[int] = None
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray]
]:
    """
    Inputs:
        m: Dict of number of samples for each modality
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
    
    Psis: dict[str, np.ndarray] = {}
    
    for name in ds.keys():
        Psis[name] = generate_Psi(
            ds[name],
            sparsities[name],
            gen_type=gen_type
        )
        
    Ys: dict = {name : None for name in structure.keys()}
    for name, axes in structure.items():
        Ys[name] = fast_ks_normal(
            [Psis[axis] for axis in axes],
            m[name],
            mean=mean
        )
        
    return Psis, Ys

def generate_sparse_invwishart_matrix(
    n: int,
    expected_nonzero: int,
    *,
    off_diagonal_scale: float = 0.9,
    size: int = 1,
    df_scale: int = 1
) -> np.ndarray:
    """
    Inputs:
        n: Number of rows/columns of output
        expected_nonzero: Number of nondiagonal nonzero entries expected
        off_diagonal_scale: Value strictly between 0 and 1 to guarantee posdefness
        size: Number of samples to return
        df_scale: How much to multiply the df parameter of invwishart, must be >= 1

    Outputs:
        (`size`, n, n) batch of sparse positive definite matrices

    Generates two sparse positive definite matrices.
    Relies on Schur Product Theorem; we create a positive definite mask matrix and
    then hadamard it with our precision matrices
    """
    
    Psi: np.ndarray
    
    # Probability to achieve desired expected value of psi nonzeros
    p: float = np.sqrt(expected_nonzero / (n**2 - n))
    
    # Generate a batch of bernoulli matrices
    b = bernoulli(p=p).rvs(size=(size, n, 1)) * np.sqrt(off_diagonal_scale)
    D = (1-b*b)*np.eye(n)
    Mask = D + b @ b.transpose([0, 2, 1])

    # Apply the mask to invwishart matrices
    Psi = invwishart.rvs(df_scale * n, np.eye(n), size=size) / (df_scale * n) * Mask
    
    # This just affects how much normalization is needed
    # TODO: Why is this needed?
    Psi /= np.trace(Psi, axis1=1, axis2=2).reshape(size, 1, 1) / n
    
    return Psi


class PrecMatMask:
    """
    The sparsity pattern for a precision matrix
    """

    def __init__(
        self,
        *,
        diagonal_scale: float = 1.1
    ):
        """
        Inputs:
            diagonal_scale:
                Value to scale the diagonal by to guarantee posdefness
        """

        self.diagonal_scale = diagonal_scale

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
        pass

class PrecMatOnes(PrecMatMask):
    """
    Matrix full of ones
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def generate(
        self,
        n: int
    ) -> np.ndarray:
        return np.ones((n, n))
    
class PrecMatIndependent(PrecMatMask):
    """
    Identity matrix
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def generate(
        self,
        n: int
    ) -> np.ndarray:
        return np.eye(n)
    
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
        n: int
    ) -> np.ndarray:
        unsymmetrized = (np.random.rand(n, n) < self.edge_probability).astype(int)

        # Symmetrize!
        unsymmetrized[np.tril_indices(n)] = 0
        symmetrized = unsymmetrized + unsymmetrized.T
        np.fill_diagonal(symmetrized, self.diagonal_scale)

        return symmetrized
    
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
        n: int
    ) -> np.ndarray:
        """
        Creates a Toeplitz matrix with p+1 nonzero off-diagonals
        (where the +1 is because diagonal should be nonzero)
        """
        diags = np.zeros(n)
        diags[:self.p+1] = 1
        return toeplitz(diags)
    
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
        n: int
    ) -> np.ndarray:
        """
        Creates a matrix with a cluster such that:
            all vertices in the cluster are connected to each other
            all vertices outside the cluster are connected to nothing
        """
        b: np.ndarray = np.random.rand(n).reshape(n, 1) < np.sqrt(self.edge_probability)
        cov_mat: np.ndarray = (b @ b.T).astype(float)
        np.fill_diagonal(cov_mat, 0)
        cov_mat += np.eye(n) * self.diagonal_scale
        return cov_mat

class PrecMatGenerator:
    """
    My new-style precision matrix generator
    """

    def __init__(
        self,
        *,
        core_type: str = "invwishart",
        mask: PrecMatMask | str = "Erdos-Renyi-Gilbert",
        scale: float = 1
    ):
        """
        There are two components to the generator:
            1. The core distribution Ψ
            2. The mask distribution M

        We then return Ψ ∘ M, where ∘ is the Hadamard product.
        By the Schur Product Theorem, this is guaranteed to be posdef
            if Ψ and M are posdef.

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

    def generate(
        self,
        n: int,
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

        mask = self.mask.generate(n)

        return self.scale * core * mask

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
    
class DatasetGenerator:
    """
    Generates potentially multimodal datasets
    """

    def __init__(
        self,
        *,
        structure: dict[Modality, tuple[Axis]],
        Ψs: Optional[dict[Axis, np.ndarray]] = None,
        generator: Optional[dict[Axis, PrecMatGenerator]] = None,
        size: Optional[dict[Axis, int]] = None,
        batch_name: str = "",
        name: str = None,
        method: Callable[[list[np.ndarray]], np.ndarray] = kron_sum_diag_fast,
    ):
        self.name = name
        self.method = method
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
        self.Ψs = Ψs
        self.generator = generator
        self.size = size

        if self.generator is None and self.Ψs is None:
            raise ValueError("Must provide either Ψs or generator")
        
        if self.size is None and self.Ψs is None:
            raise ValueError("Must provide either Ψs or size")
        
        if self.size is None:
            self.size = {
                axis: Ψ.shape[0]
                for axis, Ψ in self.Ψs.items()
            }

        if self.Ψs is None:
            self.Ψs = {
                axis: generator.generate(self.size[axis])
                for axis, generator in self.generator.items()
            }
        
    def reroll_Ψs(self):
        if self.generator is None:
            raise ValueError("Cannot reroll Ψs if generator is not provided")
        
        self.Ψs = {
            axis: generator.generate(self.size[axis])
            for axis, generator in self.generator.items()
        }

    def generate(
        self,
        m: dict[str, int]
    ) -> dict[str, np.ndarray]:
        """
        Inputs:
            m: Dict of number of samples for each modality
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
        if isinstance(m, int):
            m = {name: m for name in self.structure.keys()}

        for name, axes in self.structure.items():
            Ys[name] = fast_kronecker_normal(
                [self.Ψs[axis] for axis in axes if axis is not self.batch_name],
                m[name],
                method = self.method,
            )
            
        return Ys