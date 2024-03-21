"""
This file wraps the core functionality of the GmGM algorithm into a single function.
"""
from __future__ import annotations

# Import core functionality of GmGM
from .core.core import direct_svd, direct_left_eigenvectors
from .core.core import nonparanormal_left_eigenvectors
from .core.core import calculate_eigenvalues, calculate_eigenvectors
from .core.core import recompose_dense_precisions
from .core.preprocessing import center, clr_prost, create_gram_matrices
from .core.presparse_methods import recompose_sparse_precisions

# Used for typing
from typing import Optional, Literal
from collections.abc import Iterable
from .extras.prior import Prior
from .typing import Axis, MaybeDict
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

# For checking sparsity
import scipy.sparse as sparse

def GmGM(
    dataset: Dataset | AnnData,
    to_keep: Optional[MaybeDict[Axis, float | int]] = None,
    random_state: Optional[int] = None,
    batch_size: Optional[int] = None,
    verbose: bool = False,
    print_memory_usage: bool = False,
    # `center` parameters
    centering_method: Optional[Literal["avg-overall", "clr-prost"]] = None,
    # `create_gram_matrices` parameters
    use_nonparanormal_skeptic: bool = False,
    nonparanormal_evec_backend: Optional[Literal["COCA", "XPCA"]] = None,
    # `calculate_eigenvectors` parameters
    n_comps: Optional[int] = None,
    calculate_explained_variance: bool = True,
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
    threshold_method: MaybeSet[Literal[
        "overall",
        "overall-col-weighted",
        "rowwise",
        "rowwise-col-weighted",
        "nonsingleton-percentage"
    ]] = "overall",
    min_edges: int = 0,
    dont_recompose: Optional[set[Axis]] | bool = None,
    # from_AnnData/MuData parameters
    use_highly_variable: bool = False,
    key_added: str = "gmgm",
    use_abs_of_graph: bool = True,
    key_map: Optional[dict[Axis, Axis]] = None,
    readonly: bool = True,
    prior: Optional[dict[Axis, Prior]] = None,
    # Parameters to force a specific compute path
    _assume_sparse: bool = False,
) -> Dataset | AnnData | MuData:
    """
    Performs GmGM on the given dataset.

    `_force_sparse`: If true, will force algorithm to treat the dataset as sparse
    """
    # Convert AnnData/MuData to Dataset (if relevant)
    is_anndata: bool = AnnData is not None and isinstance(dataset, AnnData)
    is_mudata: bool = MuData is not None and isinstance(dataset, MuData)
    if is_anndata:
        _dataset = Dataset.from_AnnData(dataset, use_highly_variable=use_highly_variable)
        if prior is not None:
            _dataset.prior = prior
    elif is_mudata:
        _dataset = Dataset.from_MuData(dataset, use_highly_variable=use_highly_variable)
        if prior is not None:
            _dataset.prior = prior
    else:
        _dataset = dataset
        if prior is not None:
            raise ValueError(
                "Cannot specify `prior` if `dataset` is a Dataset.\n"
                + "Add it directly to the Dataset instead."
            )

    if readonly:
        _dataset.make_readonly()
        
    if nonparanormal_evec_backend is not None and not use_nonparanormal_skeptic:
        warnings.warn("`use_nonparanormal_skeptic` is false, so `nonparanormal_evec_backend` is ignored")

    # Save the random state
    _dataset.random_state = random_state

    # Expand `to_keep` if necessary
    # First expand if it's a single value
    if not isinstance(threshold_method, Iterable):
        to_keep = {axis: to_keep for axis in _dataset.all_axes}

    # If `dont_recompose` is a bool and true, then set it to all axes
    if dont_recompose is True:
        dont_recompose = _dataset.all_axes.copy()

    # Second expand to also contain keys of `key_map`
    if key_map is not None:
        for key, value in key_map.items():
            if value in to_keep:
                to_keep[key] = to_keep[value]
            else:
                warnings.warn(f"Key `{key}` in `key_map` not found in `to_keep`")

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


    # Check properties of dataset to find the best way to calculate eigenvectors
    unimodal: bool = len(_dataset.dataset) == 1
    sparseness = [
        _assume_sparse or sparse.issparse(_dataset.dataset[key])
        for key in _dataset.dataset.keys()
    ]
    allsparse = all(sparseness)
    anysparse = any(sparseness)

    if anysparse and not allsparse:
        warnings.warn(
            "Some axes are sparse, but not all. This is a hard case to account for,"
            + " and will likely lead to densification or slower performance (or both)."
        )

    # Assume that anything without a `ndim` attribute is a matrix
    matrix_variate: bool = all([
        not hasattr(_dataset.dataset[key], "ndim") or _dataset.dataset[key].ndim == 2
        for key in _dataset.dataset.keys()
    ])

    # If dataset is a single matrix, then we can do SVD on said matrix
    # to get the eigenvectors for both axes at once
    # (But under the nonparanormal skeptic, this trick wouldn't work)
    if unimodal and matrix_variate and n_comps is not None and not use_nonparanormal_skeptic:
        if verbose:
            print("\tby calculating SVD...")
        direct_svd(
            _dataset,
            n_comps=n_comps,
            random_state=random_state,
            calculate_explained_variance=calculate_explained_variance,
            verbose=verbose
        )
    # If dataset is a single matrix, but we want to use the nonparanormal skeptic,
    # we can do it similarly to direct_left_eigenvectors, using a rank-one update trick.
    # TODO: The trick would also work in the multi-modal case, but it is harder to
    # get sparse matrices to work with dask, so I have deferred that to later
    # TODO: This works in the tensor-variate case, but needs a more flexible library than
    # scipy.sparse, which limits us to 2D matrices
    elif unimodal and matrix_variate and n_comps is not None and use_nonparanormal_skeptic and allsparse:
        if verbose:
            print("\tby calculating left eigenvectors and applying a rank-one update...")
        nonparanormal_left_eigenvectors(
            _dataset,
            n_comps=n_comps,
            nonparanormal_evec_backend=nonparanormal_evec_backend,
            random_state=random_state,
            calculate_explained_variance=calculate_explained_variance,
            verbose=verbose,
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
            use_nonparanormal_skeptic=use_nonparanormal_skeptic,
            nonparanormal_evec_backend=nonparanormal_evec_backend,
            random_state=random_state,
            calculate_explained_variance=calculate_explained_variance,
            verbose=verbose
        )
    # If `n_comps` is not specified, we do it the old way, which is
    # an O(n^2) memory operation!
    # TODO: Update old way to still only keep min(n, m) eigenvectors/values
    # since the rest are useless (correspond to 0 eigenvalues) and
    # in fact are detrimental due to MLE existance reasons
    else:
        if verbose:
            print("\tby calculating gram matrices and then eigendecomposing...")
        warnings.warn("This is outdated, will swap to using direct_left_eigenvectors in the future")
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

        if verbose and calculate_explained_variance:
            print(f"100% explained variance, since `n_comps` was not specified")

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
    if to_keep is not None:
        if verbose:
            print("Recomposing sparse precisions...")
        recompose_sparse_precisions(
            _dataset,
            to_keep=to_keep,
            threshold_method=threshold_method,
            min_edges=min_edges,
            batch_size=batch_size,
            dont_recompose=dont_recompose
        )
    else:
        if verbose:
            print("Recomposing dense precisions as to_keep was not specified...")
        recompose_dense_precisions(
            _dataset,
            dont_recompose=dont_recompose
        )
        

    # Print memory usage (useful if directly used on `AnnData`/`MuData`)
    if print_memory_usage:
        print("Memory Usage: ")
        _dataset.print_memory_usage()

    # If was AnnData/MuData, return it
    if AnnData is not None and isinstance(dataset, AnnData):
        if verbose:
            print("Converting back to AnnData...")
        return _dataset.to_AnnData(
            key_added=key_added,
            use_abs_of_graph=use_abs_of_graph,
            key_map=key_map
        )
    elif MuData is not None and isinstance(dataset, MuData):
        if verbose:
            print("Converting back to MuData...")
        return _dataset.to_MuData(
            key_added=key_added,
            use_abs_of_graph=use_abs_of_graph,
            key_map=key_map
        )
    
    if verbose:
        print("Done!")

    if readonly:
        _dataset.unmake_readonly()
    
    # Otherwise, return a Dataset object
    return _dataset