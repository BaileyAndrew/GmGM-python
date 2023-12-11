# GmGM-python
A python package for the GmGM algorithm.  [Read the pre-print here.](https://arxiv.org/abs/2211.02920)

This is a very early version so the API is subject to change.

## Installation

```{bash}
# Pip
python -m pip install GmGM
```

Conda install coming soon.

## About

This package learns a graphical representation of every "axis" of your data.  For example, if you had a paired scRNA+scATAC multi-omics dataset, then your axes would be "genes" (columns of scRNA matrix), "axes" (columns of scATAC matrix), and "cells" (rows of both matrices).

This package works on any dataset that can be expressed as multiple tensors of arbitrary length (so multi-omics, videos, etc...).  The only restriction is that no tensor can have the same axis twice (no "genes x genes" matrix); the same axis can appear multiple times, as long as it only appears once per matrix.

## Usage

The first step is to express your dataset as a `Dataset` object.  Suppose you had a cells x genes scRNA matrix and cells x peaks scATAC matrix, then you could create a `Dataset` object like:

```{python}
from GmGM.dataset import Dataset
dataset: Dataset = Dataset(
    dataset={
        "scRNA": scRNA,
        "scATAC": scATAC
    },
    structure={
        "scRNA": ("cell", "gene"),
        "scATAC": ("cell", "peak")
    }
)
```

The basic form of the algorithm is as follows:
1) Create gram matrices (either by `center`ing and `grammifying` or using the nonparanormal skeptic)
2) Analytically `calculate_eigenvectors`
3) Iteratively `calculate_eigenvalues`
4) Recompose your precision matrices, and threshold them to be sparse (can be done in one go as `recompose_sparse_precisions` to prevent unnecessary memory use

```{python}
center(dataset)
grammify(dataset)
calculate_eigenvectors(dataset, seed=RANDOM_STATE)
calculate_eigenvalues(dataset)
recompose_sparse_precisions(
    dataset,
    to_keep=N_NEIGHBORS,
    threshold_method='rowwise-col-weighted',
    batch_size=1000
)
```

This has quadratic memory due to the computation of the Gram matrices.  When you only have a single matrix as input, you can skip this step using `direct_svd`, leading to linear memory use by directly producing the right eigenvectors from the raw data!

```{python}
center(dataset)
direct_svd(dataset, k=N_COMPONENTS, seed=RANDOM_STATE)
calculate_eigenvalues(dataset)
recompose_sparse_precisions(
    dataset,
    to_keep=N_NEIGHBORS,
    threshold_method='rowwise-col-weighted',
    batch_size=1000
)
```

All these functions are updating `dataset` in-place; the computed precision matrices are available through the `precision_matrices` attribute of `dataset`.  This is a dictionary which you index py axis name, i.e. `dataset.precision_matrices['cell']`.

## Roadmap

- [ ] Add direct support for AnnData and MuData objects (so that converson to `Dataset` is not needed)
- [ ] Stabilize API
- [ ] Add comprehensive docs
- [ ] Have `generate_data` directly generate `Dataset` objects
- [ ] Add conda distribution
- [x] Add example notebook