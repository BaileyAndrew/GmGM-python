"""
Provides the Dataset class
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import scipy.sparse as sparse

from .typing import Modality, Axis, DataTensor
from .extras.prior import Prior
from .numbafied import extract_d_values

class Dataset:
    def __init__(
        self,
        *,
        dataset: dict[Modality, DataTensor],
        structure: dict[Modality, tuple[Axis]],
        batch_axes: Optional[set[Axis]] = None
    ):
        # Type declarations of attributes
        self.dataset: dict[Modality, DataTensor]
        self.structure: dict[Modality, tuple[Axis]]

        self.batch_axes: set[Axis]
        self.presences: dict[Axis, dict[Modality, Optional[int]]]
        self.presences_batchless: dict[Axis, dict[Modality, Optional[int]]]

        self.full_sizes: dict[Modality, np.ndarray]
        self.left_sizes: dict[Modality, np.ndarray]
        self.right_sizes: dict[Modality, np.ndarray]
        self.Ks: dict[str, int]

        self.axis_sizes: dict[Axis, int]

        # Type declarations of cached attributes
        # calculated via sklearn api calls
        self.prior: dict[Axis, Prior]
        self.mean: dict[Axis, float]
        self.gram_matrices: dict[Axis, np.ndarray]
        self.es: dict[Axis, np.ndarray]
        self.evecs: dict[Axis, np.ndarray]
        self.evals: dict[Axis, np.ndarray]
        self.precision_matrices: dict[Axis, np.ndarray]

        # Input handling for default arguments
        if batch_axes is None:
            self.batch_axes = set({""})
        else:
            self.batch_axes = batch_axes

        # Set attributes
        self.dataset = dataset
        self.structure = structure
        self.gram_matrices = {}
        self.prior = {}
        self.es = {}
        self.evecs = {}
        self.evals = {}
        self.precision_matrices = {}

        # Calculate helper attributes
        self.recompute_helpers()

        # Make cached attributes for covariance subdivision
        self.cache()

    def memory_usage(self) -> tuple[int, int, int]:
        """
        Returns the memory usage of the dataset
        """
        dataset_bytes: int = 0
        aux_bytes: int = 0
        precision_bytes: int = 0

        # Keep track of dataset size
        if self.old_dataset is self.dataset:
            for tensor in self.dataset.values():
                dataset_bytes += array_bytes(tensor)
        else:
            # If parts of the dataset has been cached,
            # report the cached dataset as the uncached bits
            # are views, i.e. they do not take up any memory
            for tensor in self.old_dataset.values():
                dataset_bytes += array_bytes(tensor)

        # Add all intermediat results
        for gram_matrix in self.gram_matrices.values():
            aux_bytes += array_bytes(gram_matrix)
        for evals in self.evals.values():
            aux_bytes += array_bytes(evals)
        for evecs in self.evecs.values():
            aux_bytes += array_bytes(evecs)

        # Add output
        for precision_matrix in self.precision_matrices.values():
            precision_bytes += array_bytes(precision_matrix)

        # Potentially add cached gram matrices
        if self.gram_matrices is not self.old_gram_matrices:
            for gram_matrix in self.old_gram_matrices.values():
                aux_bytes += array_bytes(gram_matrix)

        return dataset_bytes, aux_bytes, precision_bytes
    
    def print_memory_usage(self) -> None:
        """
        Prints the memory usage of the dataset
        """
        dataset_bytes, aux_bytes, precision_bytes = self.memory_usage()
        print(f"Dataset: {dataset_bytes / 1e9:.2f} GB")
        print(f"Auxiliary: {aux_bytes / 1e9:.2f} GB")
        print(f"Output: {precision_bytes / 1e9:.2f} GB")

    def cache(self) -> None:
        self.old_dataset = self.dataset
        self.old_structure = self.structure
        self.old_all_axes = self.all_axes
        self.old_axis_sizes = self.axis_sizes
        self.old_gram_matrices = self.gram_matrices

    def uncache(self) -> None:
        self.dataset = self.old_dataset
        self.structure = self.old_structure
        self.gram_matrices = self.old_gram_matrices
        self.recompute_helpers()


    def recompute_helpers(self) -> None:
        self._calculate_axes()
        self._calculate_presences()
        self._calculate_dimensions()
        self._calculate_axis_sizes()

    def _calculate_axes(self) -> None:
        """
        Gets a set of every axis in the dataset, i.e.
        [("A", "B"), ("A", "B")]
        => {"A", "B", "C"}
        """
        self.all_axes: set[str] = {
            axis
            for axes in self.structure.values()
            for axis in axes
        } - self.batch_axes

    def _calculate_presences(self) -> None:
        """
        Returns a dictionary keyed by axis names,
        indicating which datasets the axis is present in

        We assume each axis appears at most once per dataset
        """

        self.presences: dict[Axis, dict[Modality, Optional[int]]] = {}
        self.presences_batchless: dict[Axis, dict[Modality, Optional[int]]] = {}

        for axis in self.all_axes:
            self.presences[axis] = {}
            self.presences_batchless[axis] = {}
            for modality, axes in self.structure.items():
                to_add: list[int] = [
                    idx
                    for idx, ax
                    in enumerate(axes)
                    if ax == axis
                ]

                if len(to_add) == 0:
                    self.presences[axis][modality] = None
                    self.presences_batchless[axis][modality] = None
                elif len(to_add) == 1:
                    presence: int = to_add[0]
                    batches_before_presence: int = len(
                        [
                            ax
                            for ax in axes[:presence]
                            if ax in self.batch_axes
                        ]
                    )
                    self.presences[axis][modality] = presence
                    self.presences_batchless[axis][modality] = presence - batches_before_presence
                else:
                    raise Exception("GmGM does not allow repeated axes in a dataset.")
                
    def _calculate_dimensions(self) -> None:
        """
        Get d_\forall, d_{<\ell}, and d_{>\ell}, and K for each modality,
        as defined in the paper.
        """
        self.full_sizes: dict[Modality, np.ndarray] = {}
        self.left_sizes: dict[Modality, np.ndarray] = {}
        self.right_sizes: dict[Modality, np.ndarray] = {}

        for modality, tensor in self.dataset.items():
            full_size, left_size, right_size = extract_d_values(
                tensor.shape
            )
            self.full_sizes[modality] = full_size
            self.left_sizes[modality] = left_size
            self.right_sizes[modality] = right_size

        # Get the number of dimensions for each tensor
        self.Ks: dict[str, int] = {
            modality: len(
                set(self.structure[modality]) - self.batch_axes
            )
            for modality in self.dataset.keys()
        }

    def _calculate_axis_sizes(self) -> None:
        """
        Calculates the size of each axis
        """
        self.axis_sizes: dict[Axis, int] = {}
        for axis in self.all_axes:
            self.axis_sizes[axis] = [
                tensor.shape[idx]
                for modality, tensor in self.dataset.items()
                if (idx := self.presences[axis][modality]) is not None
            ][0]

    def __repr__(self) -> str:

        # Print structure of each dataset
        dataset_str: str = "\n".join(
            f"\t{modality}: {axes}"
            for modality, axes in self.structure.items()
        )

        # Print calculation status info
        prior_str: dict[Axis, str] = {
            axis: "Prior: " + (self.prior[axis].name if axis in self.prior else "None")
            for axis in self.all_axes
        }
        gram_str: dict[Axis, str] = {
            axis: "Gram: " + ("Calculated" if axis in self.gram_matrices else "Not calculated")
            for axis in self.all_axes
        }
        eig_str: dict[Axis, str] = {
            axis: "Eig: " + ("Calculated" if axis in self.evecs else "Not calculated")
            for axis in self.all_axes
        }

        # Print the size of each axis
        size_str: str = "\n".join(
            f"\t{axis}: {size}\n\t\t"
            + "\n\t\t".join([x[axis] for x in [prior_str, gram_str, eig_str]])
            for axis, size in self.axis_sizes.items()
        )

        return f"Dataset(\n{dataset_str}\n)\nAxes(\n{size_str}\n)"
    
def array_bytes(
    arr: np.ndarray | sparse.sparray
) -> float:
    """
    Returns the number of bytes required to store
    a numpy array or sparse matrix
    """
    if isinstance(arr, sparse.spmatrix):
        raise ValueError(
            "Please use sparse arrays instead of the to-be-deprecated"
            + " sparse matrix API."
        )
    
    # Input handling for various sparse array types
    if isinstance(arr, sparse.csr_array) | isinstance(arr, sparse.csc_array):
        return (
            arr.data.nbytes
            + arr.indices.nbytes
            + arr.indptr.nbytes
        )
    elif isinstance(arr, sparse.coo_array):
        return (
            arr.data.nbytes
            + arr.row.nbytes
            + arr.col.nbytes
        )
    else:
        return arr.nbytes