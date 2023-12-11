from __future__ import annotations
from typing import TypeAlias
import numpy as np
import scipy.sparse as sparse

Axis: TypeAlias = str
Modality: TypeAlias = str
DataTensor: TypeAlias = np.ndarray
SparseArray: TypeAlias = sparse.sparray | sparse.spmatrix

"""
In python 3.12, this becomes:

type Axis = str
type Modality = str
type DataTensor = np.ndarray
type SparseArray = sparse.sparray | sparse.spmatrix
"""