"""
This file wraps the core functionality of the GmGM algorithm into a single function.
"""
from __future__ import annotations

# Import core functionality of GmGM
from .core.core import direct_svd, calculate_eigenvalues, calculate_eigenvectors
from .core.preprocessing import center, clr_prost, create_gram_matrices
from .core.presparse_methods import recompose_sparse_precisions

# Used for typing
from typing import Optional, Literal
from .typing import Axis
from .dataset import Dataset
from .extras.regularizers import Regularizer
try:
    from anndata import AnnData
except ImportError:
    AnnData = None
try:
    from mudata import MuData
except ImportError:
    MuData = None

def GmGM(
    dataset: Dataset | AnnData,
    to_keep: float | int | dict[Axis, float | int],
    random_state: Optional[int] = None,
    batch_size: Optional[int] = None,
    verbose: bool = False,
    # `center` parameters
    centering_method: Optional[Literal["avg-overall", "clr-prost"]] = None,
    # `create_gram_matrices` parameters
    use_nonparanormal_skeptic: bool = False,
    # `calculate_eigenvectors` parameters
    n_comps: Optional[int] = None,
    # `calculate_eigenvalues` parameters
    max_small_steps: int = 5,
    max_line_search_steps: int = 20,
    lr_init: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-3,
    regularizer: Optional[Regularizer] = None,
    force_posdef: bool = True,
    verbose_every: int = 100,
    always_regularize: bool = False,
    check_overstep_each_iter: bool = False,
    # `recompose_sparse_positions` parameters
    threshold_method: Literal["overall", "rowwise", "rowwise-col-weighted"] = "rowwise-col-weighted",
    # from_AnnData/MuData parameters
    use_highly_variable: bool = False,
    key_added: str = "gmgm",
):
    """
    Performs GmGM on the given dataset.
    """
    # Convert AnnData/MuData to Dataset (if relevant)
    if AnnData is not None and isinstance(dataset, AnnData):
        _dataset = Dataset.from_AnnData(dataset, use_highly_variable=use_highly_variable)
    elif MuData is not None and isinstance(dataset, MuData):
        _dataset = Dataset.from_MuData(dataset, use_highly_variable=use_highly_variable)
    else:
        _dataset = dataset

    # Save the random state
    _dataset.random_state = random_state

    # Center dataset
    if centering_method == "clr-prost":
        clr_prost(_dataset)
    elif centering_method == "avg-overall":
        center(_dataset)
    
    # Check if the dataset is unimodal, in such a case, we can skip the gram matrix calculation
    # Bringing memory usage down from O(n^2) to O(n)
    # For now requires dataset to be matrix-variate; use of hosvd may extend this?
    # In my experience it only speeds you up if you are looking at n_components!
    unimodal: bool = len(_dataset.dataset) == 1
    matrix_variate: bool = _dataset.dataset[list(_dataset.dataset.keys())[0]].ndim == 2
    if unimodal and matrix_variate and n_comps is not None:
        # Calculate eigenvectors
        direct_svd(
            _dataset,
            n_comps=n_comps,
            random_state=random_state,
        )
    # If dataset is multi-modal or is tensor-variate, we need to calculate the gram matrices
    # An O(n^2) memory operation
    else:
        # Create Gram matrices
        create_gram_matrices(
            _dataset,
            use_nonparanormal_skeptic=use_nonparanormal_skeptic,
            batch_size=batch_size
        )

        # Calculate eigenvectors
        calculate_eigenvectors(
            _dataset,
            n_comps=n_comps,
            random_state=random_state,
            verbose=verbose
        )

    # Calculate eigenvalues
    calculate_eigenvalues(
        _dataset,
        max_small_steps=max_small_steps,
        max_line_search_steps=max_line_search_steps,
        lr_init=lr_init,
        max_iter=max_iter,
        tol=tol,
        regularizer=regularizer,
        force_posdef=force_posdef,
        verbose=verbose,
        verbose_every=verbose_every,
        always_regularize=always_regularize,
        check_overstep_each_iter=check_overstep_each_iter,
    )

    # Recompose sparse precisions
    recompose_sparse_precisions(
        _dataset,
        to_keep=to_keep,
        threshold_method=threshold_method,
        batch_size=batch_size
    )

    # If was AnnData/MuData, return it
    if AnnData is not None and isinstance(dataset, AnnData):
        return _dataset.to_AnnData(key_added=key_added)
    elif MuData is not None and isinstance(dataset, MuData):
        return _dataset.to_MuData(key_added=key_added)
    
    # Otherwise, return a Dataset object
    return _dataset