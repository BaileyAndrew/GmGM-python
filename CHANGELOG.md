# Changelog

## v0.0.3 (2023/12/11)

## Dependencies
- Minimum SciPy is now 1.11, forcing minimum Python to 3.9

## Fixes
- Fixed `ImportError` for `scipy.sparse.sparray` by upgrading SciPy minimum version

## v0.0.2 (2023/12/11)

## Improvements

- Updated ReadMe

## API Changes

- Changed `GmGM` -> `calculate_eigenvalues` as that function did not represent the whole GmGM algorithm
- Changed `calculate_evalues` -> `calculate_eigenvectors` as that is more explanatory

## Fixes
- Fixed error `ImportError: cannot import name 'TypeAlias' from 'typing'` in `GmGM.typing`, which occured in Python versions less than 3.10.

## v0.0.1 (2023/12/11)

- First release