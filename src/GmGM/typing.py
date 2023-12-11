import numpy as np
import scipy.sparse as sparse

try:
    # TypeAlias only added in Python 3.10
    from typing import TypeAlias
except ImportError:
    TypeAlias = "TypeAlias"

Axis: TypeAlias = str
Modality: TypeAlias = str
DataTensor: TypeAlias = np.ndarray

"""
In python 3.12, this becomes:

type Axis = str
type Modality = str
type DataTensor = np.ndarray
"""