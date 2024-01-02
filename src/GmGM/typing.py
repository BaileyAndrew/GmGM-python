from __future__ import annotations

import numpy as np
import scipy.sparse as sparse
from typing import TypeVar, Union

try:
    # TypeAlias only added in Python 3.10
    from typing import TypeAlias
except ImportError:
    TypeAlias = "TypeAlias"

Axis: TypeAlias = str
Modality: TypeAlias = str
DataTensor: TypeAlias = np.ndarray


_key = TypeVar('_key')
_value = TypeVar('_value')

# | only added in Python 3.10 :(
MaybeDict = Union[_value, dict[_key, _value]]

"""
In python 3.12, this becomes:

type Axis = str
type Modality = str
type DataTensor = np.ndarray
type MaybeDict[Key, Value] = Value | dict[Key, Value]
"""