# Changelog

## v0.0.7 (???)

## Improvements
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