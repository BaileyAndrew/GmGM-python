# Changelog

## v0.4.1 (17/05/2024)

### Fixes
- BUG-001: `to_keep` now successfully auto-turned-into a dictionary [bug was introduced in previous version]
- BUG-002: `direct_left_eigenvectors` with `calculate_explained_variance=True` no longer crashes [dask matrix needed to be `.compute()`ed.]
- BUG-003: Passing `key_map` no longer issues faulty warnings
- BUG-004: `direct_svd` with `calculated_explained_variance=True` no longer crashes
- BUG-005: Fixed "sparse matrix is not square" error, caused when the input is represented as a sparse "matrix" instead of an "array", which affects the meaning of "**2".

### Improvements
- No longer need `verbose=True` for `Dataset.total_variance` field to be filled.
- Added a "__version__" attribute


## v0.4.0 (22/03/2024)

### API Changes
- `GmGM` now defaults to `centering_method=None`.
- Added argument `GmGM(_assume_sparse=False)`, which when True forces `GmGM` to treat the input as a sparse matrix.  This is for use when the input is sparse but cannot be detected by `scipy.sparse.issparse()`, such as `AnnData`'s experimental sparse backed formats.
- If `GmGM(threshold_method='overall', to_keep: MaybeDict[int])` or `GmGM(threshold_method='overall-col-weighted', to_keep: MaybeDict[int])`, no longer throws an error but rather converts `to_keep` to `to_keep / axis_size` and proceeds from there.
- Added argument `GmGM(calculate_explained_variance=True)` which will print out the amount of variance explained per axis if coupled with `GmGM(verbose=True)`.
- Changed default to `GmGM(threshold_method="overall")`.
- Added new thresholding method `GmGM(threshold_method="nonsingleton-percentage")`, which aims to keep a certain amount of edges such that there is a set amount of singletons left.
- Added `GmGM(min_edges=0)` to control the minimum number of edges per vertex

### Improvements
- Added `overall-col-weighted` and `nonsingleton-percentage` thresholding methods
- Rewrote `extras.regularizers.py`
- **Nonparanormal skeptic now preserves sparsity in the unimodal case.**
- Now uses `scipy.special.ndtri` rather than `scipy.stats.norm.ppf` as the former is much much faster.

## v0.3.1 (23/01/2024)

### Documentation
- Updated `examples/danio_rerio.ipynb` to showcase using `GmGM` with `MuData(axis=1)`.
- Added `examples/runtime.ipynb` to showcase runtime comparison with previous algorithms
- Added `examples/other_algs/...` to allow comparison with `TeraLasso` and `EiGLasso`.


### Improvements

- Now accepts `MuData` with `axis=1`.
- If `GmGM(key_map != None)`, then `GmGM(to_keep)` can be specified using keys from both the internal representation (`obs`, `{modality}-obs`, `var`, or `{modality}-var`) and from `key_map`.
- `GmGM.synthetic.plot_prec_recall` now removes spines from top and right of graph, for when the methods are very very good.
- `GmGM(to_keep=None)` now valid, will reconstruct whole matrix
- `GmGM` now accepts a `dont_recompose: Optional[set[Axis]]` which will tell it not to bother recomposing a certain axis
- Priors now addable (but API will change, currently attached to `Dataset`, will move to `GmGM` parameter)

### Fixes

- `GmGM.synthetic.plot_prec_recall` no longer crashes if you have less than three axes.
- `GmGM.synthetic.plot_prec_recall` uses correct colors for uncertainty fill
- `GmGM.synthetic.plot_prec_recall` now finds correct error bounds
- `binarize_matrix` now correctly produces `sparray` rather than `spmatrix` when the input is not sparse (this led to particularly nasty downstream errors!)
- `project_inv_kron_sum` formula was wrong for 3-axis and higher due to silly typo; now fixed
- `create_gram_matrices` no longer crashes when input `Dataset` contains arrays with integral dtypes

## v0.3.0 (2024/01/04)

### Improvements
- **`GmGM(use_nonparanormal_skeptic=True, n_comps!=None)` will now work, allowing linear memory use in the nonparanormal case**.  This is based on the COCA algorithm ("High Dimensional Semiparametric Scale-Invariant Principal Component Analysis" by Han and Liu), inspired by implementation given by authors of "XPCA: Extending PCA for a Combination of Discrete and Continuous Variables" (Anderson-Bergman, Kolda, Kincher-Winoto).  XPCA is a further interesting technique to investigate, and its focus on the semi-continuous case is quite promising in light of zero-inflation, but has not been included at this time.

### Fixes
- Parameters `n_power_iter`, `n_oversamples` now do as they are supposed to in `direct_svd` and `direct_left_eigenvectors`.

## v0.2.4 (2024/01/04)

### API Changes
- Added `key_map` parameter to `Dataset.from_AnnData` and `Dataset.from_MuData`, allowing the mapping of default axis names to something more useful.
- Added `readonly` parameter to `GmGM` to ensure the input matrix doesn't get written to.
- Changed default of `GmGM(centering_method=None)` to `GmGM(centering_method='avg_overall')`.
- Added new parameter `GmGM(readonly=True)` that prevents overwriting the input matrices.
- Removed `Dataset.__getitem__`
- Added new parameter `readonly=False` to `DatasetGenerator` and `PrecMatGenerator`; `GmGM.synthetic.measure_prec_recall` makes use of these parameters when generating data.

### Documentation
- Updated `example.ipynb` to show current algorithm with working nonparanormal skeptic and no longer manually create `Dataset` object, and renamed to `single_cell_transcriptomics.ipynb`.
- Improved error messages from code in `GmGM.core.presparse_methods` (no longer `AssertionError` but `ValueError`).

### Improvements
- Added `GmGM.synthetic.ZiLNMultinomial` distribution (a `ZiLNDistribution`'s outputs used as parameters to a multinomial).
- `Dataset` now has a `deepcopy` method
- Changed the way low-principal component random precision matrices are generated (still not satisfactory...)
- `GmGM.synthetic.measure_prec_recall` tries to continue chugging along even if an error gets thrown by algorithm

### Fixes
- `GmGM.synthetic.measure_prec_recall` no longer modifies generated datasets in-place
- `GmGM(use_nonparanormal_skeptic=True)` no longer divides by zero when computing correlation matrix
- Prevented accidental overwriting of input matrix
- Fixed issue with `GmGM` not always properly assigning values to input if input was a `Dataset` object rather than `AnnData`/`MuData.`

## v0.2.3 (2024/01/03)

### API Changes
- `GmGM.synthetic.plot_prec_recall` now accepts `color` and `linestyle` parameters (type `dict[AlgorithmName, str]`)

### Documentation
- `synthetic_data.ipynb` now has examples with the nonparanormal skeptic

### Fixes
- **Fixed `use_nonparanormal_skeptic=True`; before it would output nonsense!**

## v0.2.2 (2024/01/02)

### API Changes
- Removed outdated functions `generate_synthetic_dataset`, `fast_ks_normal`, `generate_Psi`, `generate_Psis`, `generate_sparse_invwishart_matrix` from `GmGM.synthetic`.
- Renamed the `method` parameter of `fast_kronecker_normal` and `DatasetGenerator` to `axis_join` and allowed it to take a string as well, instead of only allowing Callables.
- Removed unused `ell` parameter from `add_like_kron_sum`.
- Added `distribution` parameter to `DatasetGenerator` to allow user to choose whether they want their data to be normally distributed, log-normally distributed, or zero-inflated long-normally distributed.
- Changed parameter `m` of `DatasetGenerator.generate` to `num_samples`.
- `GmGM.synthetic.plot_prec_recall` now accepts multiple axes
- Removed `diagonal_scale` parameter from `GmGM.synthetic` functions; now generates negative laplacians, which are guaranteed to be posdef

### Documentation
- Updated checklist on README
- Mentioned that native MuData support is available in README
- Added type `MaybeDict` to `GmGM.typing` that allows objects to be either in a dictionary or a singleton.
- `GmGM.synthetic`'s functions now all have proper type hints
- Extended and annotated `examples/synthetic_data.ipynb`

### Improvements
- Created classes `NormalDistribution`, `LogNormalDistribution`, `ZiLNDistribution`, allowing more synthetic data options
- Added `__repr__` to `DatasetGenerator` and all affiliated classes.
- Added `__getitem__` to `Dataset`, which returns a shallow copy containing only the modalities passed into the function
- Added `n_comps` parameter to `PrecMatGenerator` to allow user to ask generator to prioritize matrices with only `n_comps` large eigenvalues.

### Fixes
- If `_estimate_sparse_gram` has `num_to_keep==0`, will no longer crash.

## v0.2.1 (2023/12/29)

### API Changes
- Moved `GmGM.generate_data` and `GmGM.validation` into new submodule `GmGM.synthetic`
- Moved `GmGM.numbafied` into `GmGM.core`
- Change default `simplify_size` of `numbafied._sum_log_sum` to 1 from 20 as this function has not been a bottleneck in the code for several months now, and higher values could lead to numeric instability (which seems to be the main factor limiting applicability to larger datasets at this point in time...).
- `GmGM.synthetic.binarize_matrices` removed and replaced with `GmGM.synthetic.binarize_precmats`

### Documentation
- Created notebook `synthetic_data.ipynb` to show off how to create and work with synthetic data.

### Improvements
- **`DatasetGenerator` now generates `Dataset` object rather than `dict[Modality, np.ndarray]`.**
- **`measure_prec_recall` now works on `Dataset` objects.**

### Fixes
- Removed some greek letters from code (`Ψ`->`Psi`) in `generate_data.py`.
- Removed some greek letters from code (`Ψ`->`Psi`, `Λ`->`Lambda`) in `validation.py.`
- Replaced occurances of `isinstance(___, float) | isinstance(___, int)` with `isinstance(___, numbers.Real)` to handle cases when the pased in value is something like `np.int16`.
- **Fixed the numba-compiled functions being very slow (slower than the python version!!)**.  This was likely due to a bug in numba; if `parallel=true` then, if `GmGM` was run several times, `project_inv_kron_sum` would get slower and slower...  Very weird!  Fixed by removing parallelism; its not needed, these functions are nowhere near the bottleneck for large datasets.
- `GmGM.synthetic.measure_prec_recall` works properly.

## v0.2.0 (2023/12/29)

### Documentation
- Added `examples/README.md`, which contains a list of common (omics-motivated) use cases
- Created notebook `single_cell_multiomics.ipynb` showing off how to use this dataset with a MuData object.

### Improvements
- `GmGM` with `clr-prost` will now raise a warning if pre-log-transformed AnnData or Mudata is passed in.
- **`GmGM` now has linear memory use in the multi-modal and tensor-variate case when assuming sparsity and a small number of `n_comps`**
- Created new function `direct_left_eigenvectors` to handle eigenvector computation in the multi-modal or tensor-variate case with small `n_comps`.
- Added method `modalities_with_axis` to `Dataset` that will give you a list of all `(location, modality)` tuples where `modality` is a modality in your dataset containing `axis` at location `location`.
- Implemented `Dataset` methods `from_MuData` and `to_MuData`
- `to/from_AnnData` now limits to highly variable `obs` as well as `var` if `use_highly_variable` is `True`.
- `to/from_AnnData` now much faster and more memory efficient for large datasets if `use_highly_variable` is `True`.
- `to/from_AnnData` can optionally not use the absolute value of the graph, with the `use_abs_of_graph` parameter.
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