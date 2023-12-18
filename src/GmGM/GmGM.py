"""
This file wraps the core functionality of the GmGM algorithm into a single function.
"""

# Import core functionality of GmGM
from .core.core import direct_svd, calculate_eigenvalues, calculate_eigenvectors
from .core.preprocessing import center, create_gram_matrices
from .core.presparse_methods import recompose_sparse_precisions

# Used for typing
from typing import Optional
from .dataset import Dataset
from .extras.regularizers import Regularizer
try:
    from AnnData import AnnData
except ImportError:
    AnnData = None
try:
    from MuData import MuData
except ImportError:
    MuData = None

def GmGM(
    dataset: Dataset | AnnData,
    random_state: Optional[int] = None,
    batch_size: Optional[int] = None,
    verbose: bool = False,
    # `center` parameters
    use_centering: bool = True,
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
):
    """
    Performs GmGM on the given dataset.
    """
    # Convert AnnData/MuData to Dataset (if relevant)
    if AnnData is not None and isinstance(dataset, AnnData):
        _dataset = Dataset.from_AnnData(dataset)
    elif MuData is not None and isinstance(dataset, MuData):
        _dataset = Dataset.from_MuData(dataset)
    else:
        _dataset = dataset

    # Center dataset
    if use_centering:
        center(_dataset)

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
    recompose_sparse_precisions(_dataset)

    # If was AnnData/MuData, return it
    if AnnData is not None and isinstance(dataset, AnnData):
        return _dataset.base
    elif MuData is not None and isinstance(dataset, MuData):
        return _dataset.base
    
    # Otherwise, return a Dataset object
    return dataset