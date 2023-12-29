# Changelog

## v0.2.0 (2023/12/29)

### Documentation
- Added `examples/README.md`, which contains a list of common (omics-motivated) use cases 

### Improvements
- `GmGM` with `clr-prost` will now raise a warning if pre-log-transformed AnnData or Mudata is passed in.
- **`GmGM` now has linear memory use in the multi-modal and tensor-variate case when assuming sparsity and a small number of `n_comps`**
- Created new function `direct_left_eigenvectors` to handle eigenvector computation in the multi-modal or tensor-variate case with small `n_comps`.
- Added method `modalities_with_axis` to `Dataset` that will give you a list of all `(location, modality)` tuples where `modality` is a modality in your dataset containing `axis` at location `location`.
- Implemented `Dataset` methods `from_MuData` and `to_MuData`
- `to/from_AnnData` now limits to highly variable `obs` as well as `var` if `use_highly_variable` is `True`.
- `to/from_AnnData` now much faster and more memory efficient for large datasets if `use_highly_variable` is `True`.
- `to/from_AnnData` can optionally not use the absolute value of the graph, with the `use_abs_of_graph` parameter.
- Created notebook `single_cell_multiomics.ipynb` showing off how to use this dataset with a MuData object.
- `GmGM` with `verbose=True` now gives more information about the computation path used.

### Fixes
- `direct_svd` now returns `X` as the type hints previously claimed.
- Updated install directions within the examples to preclude Python 3.12.
- `danio_rerio.ipynb` now respects `N_COMPONENTS` and `N_EDGES` parametrizations.
- `recompose_sparse_precisions` now properly symmetricizes its output.

## v0.1.2 (2023/12/19)

### Dependencies
- Dropped support for Python 3.12 as Numba not compatible with Python 3.12

## v0.1.1 (2023/12/18)

### Documentation
- Updated ReadMe to explain how to get results from `Dataset` object.
- Minor updates to `danio_rerio.ipynb` to demonstrate new changes.

### Improvements
- Allowed selection of multiple centering operations in `GmGM` (`avg-overall` [old way], `clr_prost`, and None)
- Added `clr_prost` method based on "A zero inflated log-normal model for inference of sparse microbial association networks" by Prost et al.
- `GmGM` now detects if your dataset is unimodal matrix-variate, and if so takes the `direct_svd` shortcut
- `direct_svd` allows `n_comps` to be `None`, which will find all singular values (although this turns out to be slower than doing it the old way...)
- `GmGM` now tries to preserve sparsity for as long as possible

## v0.1.0 (2023/12/18)

### Documentation
- Created `danio_rerio.ipynb` as an example notebook to show off both a minimal example and a way to work on more complicated multi-matrix datasets
- Updated ReadMe to use simpler API

### Improvements
- Created new function `GmGM` which wraps the standard workflow (`center` -> `create_gram_matrices` -> `calculate_eigenvectors` -> `calculate_eigenvalues` -> `recompose_precision_matrices`)
- `GmGM` accepts `AnnData` objects
- Created new class method `Dataset.from_AnnData` to create a `Dataset` from an `AnnData` object
- Created an instance method `Dataset.to_AnnData` to recover an `AnnData` object from a `Dataset` created with `AnnData`; the recovered `AnnData` will have `obsp` and `varp` attributes containing the recovered graphs
- The functions/classes `Dataset`, `GmGM`, `center`, `create_gram_matrices`, `direct_svd`, `calculate_eigenvectors`, `calculate_eigenvalues`, and `recompose_sparse_precisions` are all available as top-level imports (`from GmGM import Dataset`).
- Functions in `presparse_methods.py` can accept a batch size of `None`, which defaults to smallest of either the size of the dataset or 1000

### API Changes
- `grammify` -> `create_gram_matrices`
- `calculate_eigenvectors` now has explicit `random_state` parameter; previously was controlled by `params[seed]`
- `calculate_eigenvectors` now has explicit `n_comps` parameter; previously was controlled by `params[k]`
- `calculate_eigenvectors` no longer has a `full` parameter, as full eigendecomposition is assumed if `n_comps` is `None`.
- `direct_svd` has its `k` parameter replaced with an `n_comps` parameter to follow the ScanPy API.
- `direct_svd` has its `seed` parameter replaced with a `random_state` parameter to follow toe ScanPy API.

### Dependencies
- Added AnnData and MuData as optional dependencies

## v0.0.8 (2023/12/15)

### Improvements
- Readme gives example of how to create an environment
- `example.ipynb` has install directions at top now

## v0.0.7 (2023/12/11)

### Improvements
- Readme examples now all have import statements too

### Fixes
- Removed typo in readme

## v0.0.6 (2023/12/11)

### Fixes
- Fixed changelog indent levels
- Fixed readme syntax highlighting

## v0.0.5 (2023/12/11)

### Improvements
- Added examples to ReadMe.

## v0.0.4 (2023/12/11)

### Improvements
- `calculate_eigenvectors` no longer forces you to pick top `k` components
- Added `example.ipynb` to the GitHub repo demonstrating how to use algorithm

### Fixes
- Fixed `TypeError: unsupported operand type(s) for |: 'type' and 'type'` in `GmGM.dataset` for sparse array comparisons in older versions of python
- Fixed `TypeError: unsupported operand type(s) for |: 'type' and 'type'` in `presparse_methods` for list/int comparisons in older versions of python
- Fixed `direct_svd` `k` parameter having no effect

## v0.0.3 (2023/12/11)

### Dependencies
- Minimum SciPy is now 1.11, forcing minimum Python to 3.9

### Fixes
- Fixed `ImportError` for `scipy.sparse.sparray` by upgrading SciPy minimum version

## v0.0.2 (2023/12/11)

### Improvements

- Updated ReadMe

### API Changes

- Changed `GmGM` -> `calculate_eigenvalues` as that function did not represent the whole GmGM algorithm
- Changed `calculate_evalues` -> `calculate_eigenvectors` as that is more explanatory

### Fixes
- Fixed error `ImportError: cannot import name 'TypeAlias' from 'typing'` in `GmGM.typing`, which occured in Python versions less than 3.10.

## v0.0.1 (2023/12/11)

- First release