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
import numpy as np

def center(
    X: Dataset
) -> Dataset:
    """
    Modifies X in place
    """

    for modality in X.dataset.keys():
        X.dataset[modality] -= X.dataset[modality].mean()

    return X

def grammify(
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
