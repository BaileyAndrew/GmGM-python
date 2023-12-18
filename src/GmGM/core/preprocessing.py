"""
This file contains the methods that are meant to be used
before eigendecomposition/singular value decomposition

Specifically, this includes centering, grammifying, 
and nonparanormal-skeptic-related methods.
"""

from __future__ import annotations
from typing import Optional

from ..dataset import Dataset
from ..typing import Axis
import scipy.stats as stats
import scipy.sparse as sparse
import numpy as np

import warnings

def center(
    X: Dataset
) -> Dataset:
    """
    'centers' the dataset by subtracting each dataset's mean value
    Note: not row-mean or column-mean, but overall mean.

    Modifies X in place
    """

    for modality in X.dataset.keys():
        if sparse.issparse(X.dataset[modality]):
            warnings.warn(
                "Dataset was sparse but `avg-overall` centering will desparsify it,"
                + " consider using a different centering method."
                + "  Converting to dense array..."
            )
            X.dataset[modality] = X.dataset[modality].toarray()
        X.dataset[modality] -= X.dataset[modality].mean()

    return X

def clr_prost(
    X: Dataset,
) -> Dataset:
    """
    Takes the modified clr as described in
    "A zero inflated log-normal model for inference of sparse microbial association networks"
    by Prost et al. (2021)

    Don't use on already-logp1 transformed data

    Involves taking mean of all axes other than the first

    Modifies X in place
    """

    for modality in X.dataset.keys():
        dataset = X.dataset[modality]
        if not sparse.issparse(dataset):
            axes = tuple(range(1, len(dataset.shape)))
            dataset[dataset != 0] = np.log(dataset[dataset != 0])
            dataset -= np.sum(dataset, axis=axes, keepdims=True) / np.count_nonzero(dataset, axis=axes, keepdims=True)
        else:
            dataset = dataset.tocoo()
            dataset.data = np.log(dataset.data)
            dataset = dataset.tocsr()
            nonzero = dataset.nonzero()
            dataset[nonzero] -= (dataset.sum(axis=1).A1 / dataset.getnnz(axis=1))[nonzero[0]]

        X.dataset[modality] = dataset

def create_gram_matrices(
    X: Dataset,
    batch_size: Optional[int] = None,
    use_nonparanormal_skeptic: bool = False
) -> Dataset:
    """
    Modifies X in place
    """
    # Clear old gram matrices
    X.gram_matrices = {}

    # Calculate effective gram matrices
    num_samples: dict[Axis, int] = {}
    
    for modality, tensor in X.dataset.items():
        for idx, axis in enumerate(X.structure[modality]):
            if axis in X.batch_axes:
                continue

            if sparse.issparse(tensor):
                warnings.warn(
                    "Dataset was sparse but `create_gram_matrices` will desparsify it."
                    + "  This may be avoidable if you set the `n_comps` when calling `GmGM`."
                    + "  Converting to dense array..."
                )
                tensor = tensor.toarray()
                X.dataset[modality] = tensor

            matricized: np.ndarray = np.reshape(
                np.moveaxis(tensor, idx, 0),
                (tensor.shape[idx], -1),
                #order='F'
            )

            if axis in X.gram_matrices:
                X.gram_matrices[axis] += _grammify_core(
                    matricized,
                    batch_size,
                    use_nonparanormal_skeptic
                )
                num_samples[axis] += X.full_sizes[modality] // X.axis_sizes[axis]
            else:
                X.gram_matrices[axis] = _grammify_core(
                    matricized,
                    batch_size,
                    use_nonparanormal_skeptic
                )
                num_samples[axis] = X.full_sizes[modality] // X.axis_sizes[axis]

    # Divide gram matrices by effective number of modalities of samples seen
    for axis in X.gram_matrices.keys():
        X.gram_matrices[axis] /= num_samples[axis]

    return X

def _grammify_core(
    matricized: np.ndarray,
    batch_size: Optional[int] = None,
    use_nonparanormal_skeptic: bool = False
) -> np.ndarray:
    """
    TODO: Check if this still works
    """
    
    output: np.ndarray
    if batch_size is None:
        batch_size = matricized.shape[0]

    if use_nonparanormal_skeptic:
        for idx in range(0, matricized.shape[0] + batch_size, batch_size):
            increase = min(batch_size, matricized.shape[0] - idx)
            if idx <= 0:
                break

            matricized[idx:idx+increase] = stats.rankdata(matricized[idx:idx+increase], axis=0)
            matricized[idx:idx+increase] -= matricized[idx:idx+increase].mean(axis=0, keepdims=True)

    output = matricized @ matricized.T
    if use_nonparanormal_skeptic:
        output = np.sin(np.pi/6 * output)
    
    return output
