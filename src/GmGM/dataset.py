"""
Provides the Dataset class
"""

from __future__ import annotations

from typing import Optional, Literal
import warnings
import copy
import numpy as np
import dask.array as da
import scipy.sparse as sparse

from .typing import Modality, Axis, DataTensor
from .extras.prior import Prior
from .core.numbafied import extract_d_values

# Optional dependencies
try:
    from anndata import AnnData
    from anndata.experimental import CSRDataset, CSCDataset
except ImportError:
    AnnData = None
    CSRDataset = None
    CSCDataset = None
try:
    from mudata import MuData
except ImportError:
    MuData = None

class Dataset:
    def __init__(
        self,
        *,
        dataset: dict[Modality, DataTensor],
        structure: dict[Modality, tuple[Axis]],
        batch_axes: Optional[set[Axis]] = None,
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
        self.total_variance = {}

        # Calculate helper attributes
        self.recompute_helpers()

        # Make cached attributes for covariance subdivision
        self.cache()

        # Initialize variable that stores random state
        self.random_state: Optional[int] = None

        # Used to store tensors that have been made read-only
        self.made_readonly: set[Modality] = {}

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

        # Add all intermediate results
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

    def make_readonly(self) -> None:
        """
        Makes the dataset read-only
        """
        self.made_readonly: set[Modality] = set({})
        for modality, tensor in self.dataset.items():
            if not hasattr(tensor, "flags"):
                warnings.warn(
                    f"Trying to set {modality}'s tensor of type {type(tensor)} to read-only,"
                    + " but this dataset has no `flags` attribute.  Making a copy instead."
                )
                self.dataset[modality] = tensor.copy()
            elif tensor.flags.writeable:
                self.made_readonly.add(modality)
                tensor.flags.writeable = False

    def unmake_readonly(self) -> None:
        """
        Makes the dataset writeable
        """
        for modality, tensor in self.dataset.items():
            if not hasattr(tensor, "flags"):
                continue
            if modality in self.made_readonly:
                tensor.flags.writeable = True
        self.made_readonly = set()

    def make_writeable(self) -> None:
        """
        Makes the dataset writeable
        """
        for tensor in self.dataset.values():
            if not hasattr(tensor, "flags"):
                continue
            tensor.flags.writeable = True

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
    
    @classmethod
    def from_AnnData(
        cls,
        data: AnnData,
        use_highly_variable: bool = True,
        readonly: bool = True
    ) -> Dataset:
        if AnnData is None:
            raise ImportError("Please install AnnData to use this method.")
        matrix = data.X
        if isinstance(matrix, CSRDataset) or isinstance(matrix, CSCDataset):
            warnings.warn(
                "AnnData is relying on the experimental API.  Things may not work as"
                + " expected.  Consider saving as a Dataset."
            )
        if hasattr(matrix, "flags") and readonly:
            matrix.flags.writeable = False
        if use_highly_variable and 'highly_variable' in data.var.keys():
            matrix = matrix[:, data.var.highly_variable]

        dataset = Dataset(
            dataset={
                "AnnData" : matrix
            },
            structure={
                "AnnData": ("obs", "var")
            }
        )
        dataset.base = data
        dataset.base_use_highly_variable = use_highly_variable
        return dataset
    
    def to_AnnData(
        self,
        key_added: str = "gmgm",
        use_abs_of_graph: bool = True,
        key_map: Optional[dict[Literal["obs", "var"], Axis]] = None,
    ) -> AnnData:
        """
        `key_map` is used to map the default axis names (obs, var) to something more useful.
        """
        if AnnData is None:
            raise ImportError("Please install AnnData to use this method.")
        if not isinstance(self.base, AnnData):
            raise ValueError("This dataset was not created from an AnnData object.")
        
        if key_map is None:
            key_map = {
                "obs": "obs",
                "var": "var"
            }

        if "obs" in self.precision_matrices:
            _add_graph_to_anndata(
                self.base,
                self.base_use_highly_variable,
                key_map["obs"],
                self.base.shape[0],
                "obs",
                key_added,
                self.precision_matrices["obs"],
                self.base.obs.highly_variable \
                    if "highly_variable" in self.base.obs \
                    else None,
                use_abs_of_graph,
                self.random_state
            )
        if "var" in self.precision_matrices:
            _add_graph_to_anndata(
                self.base,
                self.base_use_highly_variable,
                key_map["var"],
                self.base.shape[0],
                "var",
                key_added,
                self.precision_matrices["var"],
                self.base.var.highly_variable \
                    if "highly_variable" in self.base.obs \
                    else None,
                use_abs_of_graph,
                self.random_state
            )

        return self.base
    
    @classmethod
    def from_MuData(
        cls,
        data: MuData,
        use_highly_variable: bool = True,
        readonly: bool = True
    ) -> Dataset:
        if MuData is None:
            raise ImportError("Please install MuData to use this method.")
        
        # Is MuData concatenating along the features axis?
        if data.axis == 0:
            matrices: dict[Modality, np.ndarray] = {}
            for modality in data.mod:
                matrix = data[modality].X
                if use_highly_variable and 'highly_variable' in data[modality].var.keys():
                    matrix = data[modality][:, data[modality].var.highly_variable].X
                if hasattr(matrix, "flags") and readonly:
                    matrix.flags.writeable = False
                matrices[modality] = matrix

            dataset = Dataset(
                dataset=matrices,
                structure={
                    modality: ("obs", f"{modality}-var")
                    for modality in data.mod
                }
            )
            dataset.base = data
            dataset.base_use_highly_variable = use_highly_variable
            return dataset
        # Is MuData concatenating along the samples axis?
        elif data.axis == 1:
            matrices: dict[Modality, np.ndarray] = {}
            for modality in data.mod:
                matrix = data[modality].X
                if use_highly_variable and 'highly_variable' in data[modality].var.keys():
                    matrix = data[modality][:, data[modality].var.highly_variable].X
                if hasattr(matrix, "flags") and readonly:
                    matrix.flags.writeable = False
                matrices[modality] = matrix

            dataset = Dataset(
                dataset=matrices,
                structure={
                    modality: (f"{modality}-obs", "var")
                    for modality in data.mod
                }
            )
            dataset.base = data
            dataset.base_use_highly_variable = use_highly_variable
            return dataset
        # Is MuData concatenating along both axes?
        elif data.axis == -1:
            raise NotImplementedError("Concatenating along both axes is not yet supported.")
        else:
            raise ValueError(f"Invalid concatenation axis {data.axis}.")
    
    def to_MuData(
        self,
        key_added: str = "gmgm",
        use_abs_of_graph: bool = True,
        key_map: Optional[dict[Axis, Axis]] = None,
    ) -> MuData:
        """
        `key_map` is used to map the default axis names (obs, var) to something more useful.
        """
        if MuData is None:
            raise ImportError("Please install MuData to use this method.")
        if not isinstance(self.base, MuData):
            raise ValueError("This dataset was not created from a MuData object.")
        
        if key_map is None:
            key_map = {}
        
        # Is MuData concatenating along the features axis?
        if self.base.axis == 0:
            obs_key = key_map["obs"] if "obs" in key_map else "obs"

            # First add the observations axis to the MuData object
            if "obs" in self.precision_matrices:
                _add_graph_to_anndata(
                    self.base,
                    self.base_use_highly_variable,
                    obs_key,
                    self.base.shape[0],
                    "obs",
                    key_added,
                    self.precision_matrices["obs"],
                    self.base.obs.highly_variable \
                        if "highly_variable" in self.base.obs \
                        else None,
                    use_abs_of_graph,
                    self.random_state
                )
            
            # Then, for each modality, add the features axis to the MuData object
            for modality in self.base.mod:
                var_key = key_map[f"{modality}-var"] if f"{modality}-var" in key_map else f"{modality}-var"
                if f"{modality}-var" in self.precision_matrices:
                    _add_graph_to_anndata(
                        self.base[modality],
                        self.base_use_highly_variable,
                        var_key,
                        self.base[modality].shape[1],
                        "var",
                        key_added,
                        self.precision_matrices[f"{modality}-var"],
                        self.base[modality].var.highly_variable \
                            if "highly_variable" in self.base[modality].var \
                            else None,
                        use_abs_of_graph,
                        self.random_state
                    )
        # Is MuData concatenating along the samples axis?
        elif self.base.axis == 1:
            var_key = key_map["var"] if "var" in key_map else "var"

            # First add the observations axis to the MuData object
            if "var" in self.precision_matrices:
                _add_graph_to_anndata(
                    self.base,
                    self.base_use_highly_variable,
                    var_key,
                    self.base.shape[0],
                    "var",
                    key_added,
                    self.precision_matrices["var"],
                    self.base.var.highly_variable \
                        if "highly_variable" in self.base.var \
                        else None,
                    use_abs_of_graph,
                    self.random_state
                )
            
            # Then, for each modality, add the features axis to the MuData object
            for modality in self.base.mod:
                obs_key = key_map[f"{modality}-obs"] if f"{modality}-obs" in key_map else f"{modality}-obs"
                if f"{modality}-obs" in self.precision_matrices:
                    _add_graph_to_anndata(
                        self.base[modality],
                        self.base_use_highly_variable,
                        obs_key,
                        self.base[modality].shape[1],
                        "obs",
                        key_added,
                        self.precision_matrices[f"{modality}-obs"],
                        self.base[modality].obs.highly_variable \
                            if "highly_variable" in self.base[modality].obs \
                            else None,
                        use_abs_of_graph,
                        self.random_state
                    )
        # Is MuData concatenating along both axes?
        elif self.base.axis == -1:
            raise NotImplementedError("Concatenating along both axes is not yet supported.")
        else:
            raise ValueError(f"Invalid concatenation axis {self.base.axis}.")
        
        return self.base
    
    def modalities_with_axis(self, axis: Axis) -> list[int, Modality]:
        """
        Returns a list of modalities that have the given axis,
        paired with the location of the axis in the modality
        """
        return [
            (list(axes).index(axis), modality)
            for modality, axes in self.structure.items()
            if axis in axes
        ]
    
    def __getitem__(self, *modalities: list[Modality]) -> Dataset:
        """
        Returns a new dataset with only the given modalities
        Is a view, not a copy (`copy.copy` preserves the pointers)
        """
        to_return = copy.copy(self)
        to_return.dataset = {
            modality: self.dataset[modality]
            for modality in modalities
        }
        to_return.structure = {
            modality: self.structure[modality]
            for modality in modalities
        }
        to_return.recompute_helpers()
        return to_return
    
    def copy(self) -> Dataset:
        """
        Returns a (shallow) copy of the dataset
        """
        return copy.copy(self)
    
    def deepcopy(self) -> Dataset:
        """
        Returns a deep copy of the dataset
        """
        return copy.deepcopy(self)
    
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
    
def _add_graph_to_anndata(
    adata: AnnData | MuData,
    use_highly_variable: bool,
    axis: Axis,
    axis_size: int,
    obs_or_var: Literal["obs", "var"],
    key_added: str,
    graph: np.ndarray | sparse.sparray | sparse.spmatrix,
    highly_variable_indices: Optional[np.ndarray] = None,
    use_abs_of_graph: bool = True,
    random_state: Optional[int] = None,
) -> AnnData | MuData:
    
    if use_abs_of_graph:
        graph = abs(graph)
    
    adata.uns[f'{axis}_neighbors_{key_added}'] = {
        'connectivities_key': f'{axis}_{key_added}_connectivities',
        'distances_key': f'{axis}_{key_added}_connectivities',
    }
    adata.uns[f'{axis}_neighbors_{key_added}']['params'] = {
        'method': 'gmgm',
    }
    if random_state is not None:
        adata.uns[f'{axis}_neighbors_{key_added}']['params']['random_state'] = random_state

    dictp: dict
    if obs_or_var == "obs":
        dictp = adata.obsp
    elif obs_or_var == "var":
        dictp = adata.varp
    else:
        raise ValueError(f"Invalid `obs_or_var`: {obs_or_var}")

    if use_highly_variable and highly_variable_indices is None:
        warnings.warn(
            "`use_highly_variable` was set to True but no highly variable indices were provided"
            + f" for axis '{axis}'.  Thus we will use all indices for this axis."
        )

    if use_highly_variable and highly_variable_indices is not None:
        # Create empty full matrix
        out_graph = sparse.coo_array((axis_size, axis_size))
        
        # Fill with highly variable connectivities
        loc = np.where(highly_variable_indices)[0]
        graph = graph.tocoo()
        out_graph.data = graph.data
        out_graph.row = loc[graph.row]
        out_graph.col = loc[graph.col]
        
        # Convert to an efficient matrix representation
        dictp[f'{axis}_{key_added}_connectivities'] = \
            out_graph.tocsr()
    else:
        dictp[f'{axis}_{key_added}_connectivities'] = graph.tocsr()

    return adata