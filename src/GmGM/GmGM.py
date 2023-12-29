"""
This file wraps the core functionality of the GmGM algorithm into a single function.
"""
from __future__ import annotations

# Import core functionality of GmGM
from .core.core import direct_svd, direct_left_eigenvectors
from .core.core import calculate_eigenvalues, calculate_eigenvectors
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

# For warnings
import warnings

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
    use_abs_of_graph: bool = True,
):
    """
    Performs GmGM on the given dataset.
    """
    # Convert AnnData/MuData to Dataset (if relevant)
    is_anndata: bool = AnnData is not None and isinstance(dataset, AnnData)
    is_mudata: bool = MuData is not None and isinstance(dataset, MuData)
    if is_anndata:
        _dataset = Dataset.from_AnnData(dataset, use_highly_variable=use_highly_variable)
    elif is_mudata:
        _dataset = Dataset.from_MuData(dataset, use_highly_variable=use_highly_variable)
    else:
        _dataset = dataset

    # Save the random state
    _dataset.random_state = random_state

    # Center dataset
    if verbose:
        print("Centering...")
    if centering_method == "clr-prost":
        if is_anndata or is_mudata:
            if 'log1p' in dataset.uns.keys():
                warnings.warn(
                    "Dataset was log1p-transformed; clr-prost expects raw compositional (such as count) data"
                )
        clr_prost(_dataset)
    elif centering_method == "avg-overall":
        center(_dataset)
    elif centering_method is None:
        pass
    else:
        raise ValueError(f"Invalid centering method: {centering_method}")
    
    # Calculate eigenvectors
    if verbose:
        print("Calculating eigenvectors...")
    # We can skip the gram matrix calculation by doing SVD directly
    # Bringing memory usage down from O(n^2) to O(n) if `n_comps`` is O(1)
    # In my experience it slows you down if we want all eigenvectors, so
    # we only do this if `n_comps`` is specified
    unimodal: bool = len(_dataset.dataset) == 1
    matrix_variate: bool = all([
        _dataset.dataset[key].ndim == 2
        for key in _dataset.dataset.keys()
    ])
    # If dataset is a single matrix, then we can do SVD on said matrix
    # to get the eigenvectors for both axes at once
    if unimodal and matrix_variate and n_comps is not None:
        if verbose:
            print("\tby calculating SVD...")
        direct_svd(
            _dataset,
            n_comps=n_comps,
            random_state=random_state,
        )
    # If dataset is multi-modal or tensor-variate, we can find the left
    # eigenvectors of the concatenation of the matricization of each modality
    # on a given axis
    elif n_comps is not None:
        if verbose:
            print("\tby calculating left eigenvectors of concatenated matricizations...")
        direct_left_eigenvectors(
            _dataset,
            n_comps=n_comps,
            random_state=random_state,
        )
    # If dataset is multi-modal or is tensor-variate, we need to calculate the gram matrices
    # An O(n^2) memory operation
    else:
        if verbose:
            print("\tby calculating gram matrices and then eigendecomposing...")
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
    if verbose:
        print("Calculating eigenvalues...")
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
    if verbose:
        print("Recomposing sparse precisions...")
    recompose_sparse_precisions(
        _dataset,
        to_keep=to_keep,
        threshold_method=threshold_method,
        batch_size=batch_size
    )

    # If was AnnData/MuData, return it
    if AnnData is not None and isinstance(dataset, AnnData):
        if verbose:
            print("Converting back to AnnData...")
        return _dataset.to_AnnData(key_added=key_added, use_abs_of_graph=use_abs_of_graph)
    elif MuData is not None and isinstance(dataset, MuData):
        if verbose:
            print("Converting back to MuData...")
        return _dataset.to_MuData(key_added=key_added, use_abs_of_graph=use_abs_of_graph)
    
    if verbose:
        print("Done!")
    
    # Otherwise, return a Dataset object
    return _dataset