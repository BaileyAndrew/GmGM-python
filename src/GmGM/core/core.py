"""
This file contains the core methods of GmGM,
specifically Theorems 1 and 2 of the ArXiv paperß
"""

from __future__ import annotations
from typing import Optional, Literal

from ..dataset import Dataset
from ..typing import Axis, Modality
from .numbafied import project_inv_kron_sum, sum_log_sum
import numpy as np
import dask.array as da
import scipy.sparse as sparse
import scipy.stats as stats
import scipy.special as special

import warnings

# TODO: Figure out what to do with these
from ..extras.regularizers import Regularizer
from ..extras.prior import Prior

def direct_left_eigenvectors(
    X: Dataset,
    n_comps: Optional[int] = None,
    n_power_iter: int = 4,
    n_oversamples: int = 100,
    use_nonparanormal_skeptic: bool = False,
    nonparanormal_evec_backend: Optional[Literal["COCA", "XPCA"]] = None,
    random_state: Optional[int] = None,
    calculate_explained_variance: bool = False,
    verbose: bool = False
) -> Dataset:
    """
    No assumptions on `X` but works best when `n_comps` is specified
    """

    if n_comps is None:
        raise NotImplementedError("Need to specify `n_comps` for now")
    
    for axis in X.all_axes:
        if axis in X.batch_axes:
            continue
        to_concat: list[da.Array] = []
        for location, modality in X.modalities_with_axis(axis):
            tensor = X.dataset[modality]
            if sparse.issparse(tensor):
                warnings.warn(
                    "Dataset was sparse but `direct_left_eigenvectors` will desparsify it."
                    + "  This will change eventually."
                )
                tensor = tensor.toarray()
            matricized: da.array = da.from_array(np.reshape(
                np.moveaxis(tensor, location, 0),
                (tensor.shape[location], -1),
            ))
            to_concat.append(matricized)
        full_matricized: da.array = da.concatenate(to_concat, axis=1)

        if use_nonparanormal_skeptic:
            if nonparanormal_evec_backend is None:
                warnings.warn("`nonparanormal_evec_backend` unspecified, defaulting to `COCA`")
                nonparanormal_evec_backend = "COCA"
            if nonparanormal_evec_backend == "COCA":
                # Convert data to uniform distribution
                # (Need to add 1 because otherwise 1 would be maximum value and this maps to infinity)
                full_matricized = da.apply_along_axis(
                    lambda a: stats.rankdata(a, method="average"),
                    0,
                    full_matricized,
                    shape=(full_matricized.shape[0],)
                )
                full_matricized = full_matricized / (full_matricized.shape[0]+1)

                # Convert uniform distribution to normal distribution
                full_matricized = full_matricized.map_blocks(special.ndtri)

            elif nonparanormal_evec_backend == "XPCA":
                raise NotImplementedError("XPCA backend not yet implemented")
            else:
                raise ValueError(f"Unknown `nonparanormal_evec_backend`: {nonparanormal_evec_backend}")

        V_1, Lambda, _ = da.linalg.svd_compressed(
            full_matricized,
            k=n_comps,
            compute=True,
            n_power_iter=n_power_iter,
            n_oversamples=n_oversamples,
            seed=random_state
        )
        Lambda = (Lambda**2).compute()
        V_1 = V_1.compute()
        X.evecs[axis] = V_1
        X.es[axis] = Lambda

        if verbose and calculate_explained_variance:
            total_variance = (full_matricized**2).sum()
            explained_variance = X.es[axis].sum() / total_variance
            X.total_variance[axis] = total_variance
            print(f"\t\tExplained variance for {axis=}: {explained_variance:.4%}")

    return X

def nonparanormal_left_eigenvectors(
    X: Dataset,
    n_comps: Optional[int] = None,
    nonparanormal_evec_backend: Optional[Literal["COCA", "XPCA"]] = None,
    random_state: Optional[int] = None,
    verbose: bool = False,
    calculate_explained_variance: bool = False
) -> Dataset:
    """
    Similar to `direct_left_eigenvectors` but does not densify the dataset
    when it is sparse.

    TODO: Currently only works when dataset is unimodal
    """

    if nonparanormal_evec_backend is None:
        warnings.warn("`nonparanormal_evec_backend` unspecified, defaulting to `COCA`")
        nonparanormal_evec_backend = "COCA"

    if nonparanormal_evec_backend == "XPCA":
        raise NotImplementedError("XPCA backend not yet implemented")

    unimodal = len(X.dataset) == 1

    if not unimodal:
        raise NotImplementedError(
            "Currently only implemented for unimodal datasets,"
            + " although there is no theoretical barrier to its extension."
        )
    
    if unimodal:
        dataset = X.dataset[list(X.dataset.keys())[0]]
        axes = list(X.structure.values())[0]

        for i, axis in enumerate(axes):
            if axis in X.batch_axes:
                continue

            if i >= 2:
                raise NotImplementedError(
                    "Currently only implemented for unimodal datasets,"
                    + " as scipy.sparse does not support higher-dimensional arrays."
                )
            
            if verbose:
                print(f"\t\tComputing sparse normal map for {axis=}...")

            if i == 0:
                A, b, total_variance = _sparse_normal_map(
                    dataset,
                    calculate_explained_variance=calculate_explained_variance
                )
            elif i == 1:
                A, b, total_variance = _sparse_normal_map(
                    dataset.T,
                    calculate_explained_variance=calculate_explained_variance
                )

            if verbose:
                print("\t\t...Done computing sparse normal map")

            # Compute the SVD of A
            V_1, Lambda, V_2T = sparse.linalg.svds(
                A,
                k=n_comps,
                random_state=random_state
            )
            V_2 = V_2T.T

            # Rank-one update to get SVD of A + 1@b.T
            V_1, Lambda, V_2 = _svd_rank_one_update(V_1, Lambda, V_2, b)

            X.evecs[axis] = V_1
            X.es[axis] = Lambda ** 2

            if verbose and calculate_explained_variance:
                explained_variance = X.es[axis].sum() / total_variance
                X.total_variance[axis] = total_variance
                print(f"\t\tExplained variance for {axis=}: {explained_variance:.4%}")

    return X
        


def direct_svd(
    X: Dataset,
    n_comps: Optional[int] = None,
    n_power_iter: int = 4,
    n_oversamples: int = 100,
    random_state: Optional[int] = None,
    calculate_explained_variance: bool = False,
    verbose: bool = False
) -> Dataset:
    """
    Assumes `X` is a single matrix

    Split into four cases:
    (sparse, truncated)
    (sparse, not truncated)
    (dense, truncated)
    (dense, not truncated)
    """
    if len(X.dataset) != 1:
        raise ValueError("Dataset must be a single matrix")
    
    dataset = X.dataset[list(X.dataset.keys())[0]]
    if n_comps is not None:
        if sparse.issparse(dataset):
            V_1, Lambda, V_2 = sparse.linalg.svds(
                dataset,
                k=n_comps,
                random_state=random_state
            )
            Lambda = Lambda**2
            V_2 = V_2.T
        else:
            if not isinstance(dataset, da.Array):
                dataset = da.from_array(dataset)
            V_1, Lambda, V_2 = da.linalg.svd_compressed(
                dataset,
                k=n_comps,
                compute=True,
                n_power_iter=n_power_iter,
                n_oversamples=n_oversamples,
                seed=random_state
            )
            Lambda = (Lambda**2).compute()
            V_1 = V_1.compute()
            V_2 = V_2.T.compute()
    else:
        if sparse.issparse(dataset):
            warnings.warn(
                "Dataset was sparse but need density to compute all svds,"
                + " consider using the `n_comps` parameter."
                + "  Converting to dense array..."
            )
            dataset = dataset.toarray()
        if not isinstance(dataset, da.Array):
            dataset = da.from_array(dataset)
        V_1, Lambda, V_2 = da.linalg.svd(dataset)
        Lambda = (Lambda**2).compute()
        V_1 = V_1.compute()
        V_2 = V_2.T.compute()

    first_axis, second_axis = list(X.structure.values())[0]

    X.evecs[first_axis] = V_1
    X.evecs[second_axis] = V_2
    X.es[first_axis] = Lambda
    X.es[second_axis] = Lambda

    if verbose and calculate_explained_variance:
        total_variance = (dataset**2).sum()
        explained_variance = X.es[first_axis].sum() / total_variance
        print(f"\t\tExplained variance for {first_axis=}: {explained_variance:.4%}")
        explained_variance = X.es[second_axis].sum() / total_variance
        print(f"\t\tExplained variance for {second_axis=}: {explained_variance:.4%}")
        print(f"\t\t\t(These values should be approximately equal)")
        X.total_variance[first_axis] = total_variance
        X.total_variance[second_axis] = total_variance
    return X

def calculate_eigenvectors(
    X: Dataset,
    verbose: bool = False,
    n_comps: Optional[int] = None,
    random_state: Optional[int] = None,
    **params
) -> Dataset:
    # Initialize the gram matrices
    grams: dict[Axis, np.ndarray] = {}
    for axis in X.all_axes:
        grams[axis] = X.gram_matrices[axis]

    # Update the gram matrices that have priors
    for axis, prior in X.prior.items():
        grams[axis] = prior.process_gram(
            X.gram_matrices[axis]
        )

    # Next, calculate the eigenvalues and eigenvectors
    es: dict[Axis, np.ndarray] = {}
    evecs: dict[Axis, np.ndarray] = {}

    for axis, gram_matrix in grams.items():
        if verbose:
            print(f"Calculating eigenvectors for {axis=}")
        if n_comps is not None:
            if not isinstance(gram_matrix, da.Array):
                gram_matrix = da.from_array(gram_matrix)
            _, s, eigenvectors = da.linalg.svd_compressed(gram_matrix, seed=random_state, k=n_comps, **params)
            eigenvalues = s**2
            eigenvectors = eigenvectors.compute().T
            eigenvalues = eigenvalues.compute()
        else:
            if isinstance(gram_matrix, da.Array):
                gram_matrix = gram_matrix.compute()
            eigenvalues, eigenvectors = np.linalg.eigh(gram_matrix)

        es[axis] = eigenvalues
        evecs[axis] = eigenvectors

    # Convert eigenvectors to fortran order
    # This will make large multiplications faster, it seems!
    evecs = {
        axis: np.asfortranarray(evec)
        for axis, evec in evecs.items()
    }

    """
    When the trace term is large, we expect the eigenvalues to be small, as naive
    Gram matrix inversion would have the eigenvalues be the inverse of the trace term.
    This puts us close to the boundary of the positive definite cone, which is why
    it performs poorly - we have to tread to lightly and end up "converging" before
    we actually reach the optimum.

    Downscaling the trace term to be around 1 will mean the reciprocal also stays around 1,
    which keeps us away from the boundary.
    """
    es = {
        axis: es[axis] / np.linalg.norm(es[axis])
        for axis in X.all_axes
    }

    # Store the eigenvalues and eigenvectors
    X.es = es
    X.evecs = evecs

    return X

def calculate_eigenvalues(
    X: Dataset,
    *,
    max_small_steps: int = 5,
    max_line_search_steps: int = 20,
    lr_init: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-3,
    regularizer: Optional[Regularizer] = None,
    force_posdef: bool = True,
    verbose: bool = False,
    verbose_every: int = 100,
    always_regularize: bool = False,
    check_overstep_each_iter: bool = False,
) -> Dataset:

    # Initialize looping variables
    num_small_steps: int = 0
    lr_t: float = lr_init
    prev_err: float = np.inf
    regularizing: bool = always_regularize and not regularizer is None

    # Eigenvalues of the mle, keyed by axis
    X.evals: dict[Axis, np.ndarray] = {
        axis: np.ones_like(e)
        for axis, e in X.es.items()
    }

    # Changes in eigenvalues in current iteration
    # keyed by axis
    diffs: dict[Axis, np.ndarray] = {
        axis: np.zeros_like(e)
        for axis, e in X.es.items()
    }

    # Converge to eigenvalue MLE
    for i in range(max_iter):

        # Compute MLE gradient
        for axis, locations in X.presences_batchless.items():
            assert axis not in X.batch_axes, \
                "Batch axes should not be in presences_batchless"
            diffs[axis] = X.es[axis].copy()

            for modality, ell in locations.items():
                if ell is None:
                    continue
                # Note to self: often it seems project_inv_kron_sum
                # is irrelevant, because it doesn't affect much...
                # Presumably the log determinant is much smaller in effect
                # than the trace...
                # Could be worth investigating how often this is the case
                # and what this says about multi-axis methods
                diffs[axis] -= project_inv_kron_sum(
                    X.evals,
                    X.structure,
                    modality,
                    X.batch_axes,
                    X.Ks[modality],
                )[ell]

                if axis in X.prior:
                    diffs[axis] += X.prior[axis].process_gradient(
                        X.evals[axis]
                    )

        # Add regularization if necessary
        # if regularizing:
        #     diffs = regularizer.prox(diffs, X.evecs)

        # Backtracking line search
        line_search_gave_up: bool = False
        lr_t: float = lr_init
        for line_step in range(max_line_search_steps):
            # Decrease step size each time
            # (`line_step` starts at 0, i.e. no decrease)
            step_lr: float = lr_t / 10**line_step
            
            for axis in X.all_axes:
                X.evals[axis] -= step_lr * diffs[axis]

            # Since all tuplets of eigenvalues
            # get summed together within each dataset,
            # the minimum final eigenvalue is 
            # the sum of minimum axis eigenvalues
            if not force_posdef:
                minimum_diag: float = min(
                    sum(
                        X.evals[axis].min()
                        for axis in axes
                        if axis not in X.batch_axes
                    )
                    for axes in X.structure.values()
                )
            # Or we can enforce individual posdefness!
            else:
                minimum_diag = min(
                min(
                    X.evals[axis].min()
                    for axis in axes
                    if axis not in X.batch_axes
                )
                for axes in X.structure.values()
                )

            # If the minimum eigenvalue is less than zero
            # have we left the positive definite space we desire
            # to stay in, so we will have to backtrack
            if minimum_diag <= 1e-8:
                for axis in X.all_axes:
                    X.evals[axis] += step_lr * diffs[axis]
                continue
            

            # Check if error has gotten worse
            if check_overstep_each_iter:
                log_err: float = _get_log_err(
                    X.evals,
                    X.structure
                )
                trace_err: float = _get_trace_err(
                    X.es,
                    X.evals,
                    X.all_axes
                )
                reg_err: float = _get_reg_err(
                    X.evals,
                    X.evecs,
                    regularizing,
                    regularizer
                )
                err: float = log_err + trace_err + reg_err
                if err > prev_err:
                    for axis in X.all_axes:
                        X.evals[axis] += step_lr * diffs[axis]
                    continue
            
            # If it got here, we have a good step size
            break
        else:
            # Did not find a good step size
            if verbose:
                print(f"@{i}: {prev_err} - Line Search Gave Up!")
            line_search_gave_up = True
            num_small_steps = max_line_search_steps + 1

        # Calculate the error
        if not check_overstep_each_iter:
            log_err: float = _get_log_err(
                X.evals,
                X.structure
            )
            trace_err: float = _get_trace_err(
                X.es,
                X.evals,
                X.all_axes
            )
            reg_err: float = _get_reg_err(
                X.evals,
                X.evecs,
                regularizing,
                regularizer
            )
            err: float = log_err + trace_err + reg_err

        # Apply proximal operator
        if regularizing:
            X.evals = regularizer.prox(X.evals, X.evecs, step_lr)

        # Calculate the change in error and
        # whether or not we can consider
        # ourselves to be converged
        err_diff: float = np.abs(prev_err - err)
        prev_err: float = err
        if err_diff/np.abs(err) < tol or line_search_gave_up:
            num_small_steps += 1
            if num_small_steps >= max_small_steps:
                if verbose:
                    print(f"Converged! (@{i}: {err})")
                if not regularizing and regularizer is not None:
                    if verbose:
                        print("Regularizing!")
                    tol /= 10
                    regularizing = True
                    num_small_steps = 0
                else:
                    break
        else:
            num_small_steps = 0

        if verbose:
            if i % verbose_every == 0:
                print(f"@{i}: {err} ({log_err} + {trace_err} + {reg_err}) ∆{err_diff / np.abs(err)}")
    else:
        # This triggers if we don't break out of the loop
        if verbose:
            print("Did not converge!")

    return X

def recompose_dense_precisions(
    X: Dataset,
    dont_recompose: Optional[set[Axis]] = None
) -> Dataset:
    """
    Recomposes the dense precision matrices
    """
    if dont_recompose is None:
        dont_recompose = set({})
    for axis in X.all_axes:
        if axis in dont_recompose:
            continue
        X.precision_matrices[axis] = (
            (X.evecs[axis] * X.evals[axis])
            @ X.evecs[axis].T
        )
    return X
    
def _get_log_err(
    evals: dict[Axis, np.ndarray],
    structure: dict[Modality, tuple[Axis]],
) -> float:
    """
    Calculates log-determinant portion of the error
    """
    log_err: float = 0
    for _, axes in structure.items():
        log_err -= sum_log_sum(
            *[
                eval
                for axis, eval
                in evals.items()
                if axis in axes
            ]
        )
    return log_err

def _get_trace_err(
    es: dict[Axis, np.ndarray],
    evals: dict[Axis, np.ndarray],
    all_axes: set[Axis]
) -> float:
    """
    Returns the trace portion of the error
    """
    trace_err: float = sum(
                es[axis]
                @ evals[axis]
                for axis in all_axes
            )
    return trace_err

def _get_reg_err(
    evals: dict[Axis, np.ndarray],
    evecs: dict[Axis, np.ndarray],
    regularizing: bool,
    regularizer: Regularizer
) -> float:
    """
    Returns the regularization penalty
    portion of the error
    """
    if regularizing:
        reg_err: float = regularizer.loss(
            evals,
            evecs,
        )
    else:
        reg_err: float = 0
    return reg_err

def _svd_rank_one_update(U, S, V, a):
    """
    Computes SVD of USV^T + a1^T via rank one update.
    """
    p, r = U.shape
    q, _ = V.shape

    # Make sure this constitutes a valid SVD
    assert S.shape == (r,)
    assert a.shape == (q,)
    assert U.shape == (p, r)
    assert V.shape == (q, r)

    # Orthogonal projection vectors
    m = U.sum(axis=0)
    P = 1 - U @ m
    R_a = np.linalg.norm(P)
    P /= R_a

    n = V.T @ a
    Q = a - V @ n
    R_b = np.linalg.norm(Q)
    Q /= R_b


    # Create the K that should be eigendecomped
    K1 = np.zeros((r+1, r+1))
    np.fill_diagonal(K1[:r, :r], S)
    K2 = (
        np.concatenate((m, np.array([R_a]))).reshape(-1, 1)
        @ np.concatenate((n, np.array([R_b]))).reshape(1, -1)
    )
    K = K1 + K2

    # Inner eigendecomp
    Up, Sf, VpT = np.linalg.svd(K)
    Vp = VpT.T

    # Results
    Uf = np.hstack((U, P.reshape(-1, 1))) @ Up
    Vf = np.hstack((V, Q.reshape(-1, 1))) @ Vp
    
    return Uf, Sf, Vf

def _sparse_normal_map(
    X: sparse.sparray,
    method: Literal["min", "average", "max"] = "average",
    calculate_explained_variance: bool = False
) -> tuple[sparse.sparray, np.ndarray, float]:
    """
    Given a sparse matrix X (p by q), maps it to a normal distribution.
    To preserve sparsity, we return the output expressed as a sum:
    A + zeromaps @ np.ones(q)^T
    where A has the same sparsity pattern as X, and zeromaps is a p-vector
    containing the value (per-row) that zero was mapped to by the transformation.

    This enables us to operate on a sparse matrix A, and use zeromaps later for
    rank-one updates of those operations.  This helps avoid the need to densify.

    Returns A, zeromaps, total_variance
    (if calculate_explained_variance is False, total_variance = 0)
    """
    p, q = X.shape
    total_variance = 0

    # Have to copy anyways, so we might as well convert
    # to csc format for efficient column slicing
    # Note that once you start running out of memory,
    # this line will become the bottleneck
    A = X.tocsc(copy=True)

    zeromaps = np.zeros(q)
    for i in range(q):

        # Gets ith column, just like Y = A[:, i]
        # but this is a particularly efficient way of doing this
        Y_nonzero = A.data[A.indptr[i]:A.indptr[i+1]]

        # Get number of zeros in Y
        num_zeros = p - Y_nonzero.size

        cur = stats.rankdata(Y_nonzero, axis=0, method=method)
        
        if num_zeros > 0:
            if method == "min" or method == "dense":
                rank = (Y_nonzero < 0).sum() + 1
            elif method == "max":
                rank = p - (Y_nonzero > 0).sum()
            elif method == "average":
                Y_subzero = (Y_nonzero < 0).sum()
                rank_min = Y_subzero + 1
                rank_max = (p - Y_subzero - Y_nonzero.size)
                rank = (rank_min + rank_max) / 2
            elif method == "ordinal":
                # This is not possible to implement as we need all zeros
                # to be mapped to the same value
                raise NotImplementedError("Ordinal method not possible to implement")
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # special.ndtri == stats.norm.ppf, but ndtri is much faster
            # as it contains none of norm's boilerplate
            zeromaps[i] = special.ndtri(rank / (p+1))

            # For every cur that is a larger rank than 0's rank, add the number of nonzeros to the rank
            cur[cur > rank] += num_zeros

        # Now map to normal distribution
        cur = special.ndtri(cur / (p+1))

        if calculate_explained_variance:
            total_variance += (cur**2).sum()
            total_variance += (zeromaps[i]**2) * num_zeros

        # And subtract the mapped zero
        cur -= zeromaps[i]

        # This is how to get indices in ith column for csc
        # Looks complicated, but is same as A[:, i].data = cur
        # but faster
        A.data[A.indptr[i]:A.indptr[i+1]] = cur

    return A, zeromaps, total_variance