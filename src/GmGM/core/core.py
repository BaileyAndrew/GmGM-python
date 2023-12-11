"""
This file contains the core methods of GmGM,
specifically Theorems 1 and 2 of the ArXiv paperß
"""

from __future__ import annotations
from typing import Optional

from ..dataset import Dataset
from ..typing import Axis, Modality
from ..numbafied import project_inv_kron_sum, sum_log_sum
import numpy as np
import dask.array as da

# TODO: Figure out what to do with these
from ..extras.regularizers import Regularizer
from ..extras.prior import Prior

def direct_svd(
    X: Dataset,
    k: int = 200,
    n_power_iter: int = 4,
    n_over_samples: int = 100,
    seed: Optional[int] = None
) -> Dataset:
    """
    Assumes Dataset is a single matrix
    """
    if len(X.dataset) != 1:
        raise ValueError("Dataset must be a single matrix")
    
    V_1, Lambda, V_2 = da.linalg.svd_compressed(
        X.dataset[list(X.dataset.keys())[0]],
        k=200,
        compute=True,
        n_power_iter=4,
        n_oversamples=100,
        seed=seed
    )
    Lambda = (Lambda**2).compute()
    V_1 = V_1.compute()
    V_2 = V_2.T.compute()

    first_axis, second_axis = list(X.structure.values())[0]

    X.evecs[first_axis] = V_1
    X.evecs[second_axis] = V_2
    X.es[first_axis] = Lambda
    X.es[second_axis] = Lambda

def calculate_eigenvectors(
    X: Dataset,
    verbose: bool = False,
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
            print(f"Calculating eigenvalues for {axis=}")
        if not isinstance(gram_matrix, da.Array):
            gram_matrix = da.from_array(gram_matrix)
        _, s, eigenvectors = da.linalg.svd_compressed(gram_matrix, **params)
        eigenvalues = s**2
        eigenvectors = eigenvectors.compute().T
        eigenvalues = eigenvalues.compute()

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
                # can be removed, because it doesn't affect much...
                # Presumably the log determinant is much smaller in effect
                # than the trace...
                # Could be worth investigating how often this is the case
                diffs[axis] -= project_inv_kron_sum(
                    X.evals,
                    X.structure,
                    modality,
                    X.batch_axes,
                    X.Ks[modality],
                )[ell]

                for axis, prior in X.prior.items():
                    diffs[axis] += prior.process_gradient(
                        X.evals[axis]
                    )

        # Add regularization if necessary
        # Only activates after converging to the MLE
        if regularizing:
            regs: dict[str, np.ndarray] = regularizer.grad(
                X.evals,
                X.evecs,
            )
            for axis in X.all_axes:
                diffs[axis] += regs[axis]

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