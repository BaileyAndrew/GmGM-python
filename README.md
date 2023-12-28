# GmGM-python
A python package for the GmGM algorithm.  [Read the pre-print here.](https://arxiv.org/abs/2211.02920)

This is a very early version so the API is subject to change.

## Installation

We recommend installing this package in a conda environment, by first running:
```bash
conda create -n {YOUR ENVIRONMENT NAME} "python>=3.9,<3.12"
conda activate {YOUR ENVIROMENT NAME}
```

Afterwards you can install it via pip.

```bash
# Pip
python -m pip install GmGM
```

Conda install coming soon.

## About

This package learns a graphical representation of every "axis" of your data.  For example, if you had a paired scRNA+scATAC multi-omics dataset, then your axes would be "genes" (columns of scRNA matrix), "axes" (columns of scATAC matrix), and "cells" (rows of both matrices).

This package works on any dataset that can be expressed as multiple tensors of arbitrary length (so multi-omics, videos, etc...).  The only restriction is that no tensor can have the same axis twice (no "genes x genes" matrix); the same axis can appear multiple times, as long as it only appears once per matrix.

## Usage

For an example, we recommend looking at the `danio_rerio.ipynb` notebook.

### With AnnData

If you already have your data stored as an AnnData object, GmGM can be used directly.  Suppose you had a single-cell RNA sequencing dataset `scRNA`.

```python
GmGM(
    scRNA,
    to_keep={
        "obs": 10,
        "var": 10,
    }
)
```

`"obs": 10` tells the algorithm to keep 10 edges per cell (the 'obs' axis of AnnData) and `"var": 10` tells the algorithm to keep 10 edges per gene (the 'var' axis of AnnData).

This modifies the AnnData object in place, storing the resultant graphs in `scRNA.obsp["obs_gmgm_connectivities"]` and `scRNA.varp["var_gmgm_connectivities"]`.


### With MuData

Native support for MuData is coming soon.

### General Usage (i.e. Without AnnData/MuData)

The first step is to express your dataset as a `Dataset` object.  Suppose you had a cells x genes scRNA matrix and cells x peaks scATAC matrix, then you could create a `Dataset` object like:

```python
from GmGM import Dataset
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

Running GmGM is then as simple as:

```python
GmGM(
    dataset,
    to_keep={
        "cell": 10,
        "gene": 10,
        "peak": 10
    }
)
```

`to_keep` tells the algorithm how many edges to keep per cell/gene/peak.

The final results are stored in `dataset.precision_matrices["cell"]`, `dataset.precision_matrices["gene"]`, and `dataset.precision_matrices["peak"]`, respectively.


## Roadmap

- [x] Add direct support for AnnData objects
- [ ] Add direct support for MuData objects (so that converson to `Dataset` is not needed)
- [ ] Stabilize API
- [ ] Add comprehensive docs
- [ ] Have `generate_data` directly generate `Dataset` objects
- [ ] Add conda distribution
- [x] Add example notebook
- [ ] Make sure regularizers still work
- [ ] Make sure priors still work
- [ ] Make sure covariance thresholding trick still works
- [ ] Add unit tests
- [ ] Allow forcing subset of axes to have given precision matrices