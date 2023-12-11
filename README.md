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

The first step is to express your dataset as a "Dataset" object.

## Roadmap

- [ ] Add direct support for AnnData and MuData objects (so that converson to "Dataset" is not needed)
- [ ] Stabilize API
- [ ] Add comprehensive docs