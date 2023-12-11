"""
This file contains the methods corresponding to the covariance
subdivision trick (Theorem 3 in the ArXiv paper for GmGM)
"""

from __future__ import annotations
from typing import Optional

from ..typing import DataTensor
from ..dataset import Dataset, Modality, Axis
import scipy.sparse as sparse
import numpy as np

# TODO: Replace this dependency with sparse.csgraph
import igraph as ig

# TODO: Create context managers for these

def subdivide_covariance(
    X: Dataset,
    threshold: Optional[dict[Axis, float] | float],
    *,
    axis_separator: str = "||",
    min_component_size: int = 2,
    verbose: bool = False,
) -> Dataset:
    """
    Uses Theorem 3 in the GmGM ArXiv paper to reduce the problem size
    """
    X.cache()

    # Do covariance partitioning
    components_map: dict[Axis, ig.VertexClustering] = {}
    to_skip: dict[Axis, set[int]] = {axis: set({}) for axis in X.all_axes}
    new_dataset: dict[Modality, DataTensor] = dict({})
    new_grams: dict[Axis, np.ndarray] = dict({})
    new_structure: dict[Modality, tuple[Axis, ...]] = dict({})

    if threshold is None:
        threshed_gram: dict[Axis, np.ndarray] = X.gram_matrices
    elif isinstance(threshold, float):
        threshed_gram: dict[Axis, np.ndarray] = {
            axis: Threshold.shrink_axis(X.gram_matrices[axis], threshold, safe=True)
            for axis in X.gram_matrices.keys()
        }
    else:
        threshed_gram: dict[Axis, np.ndarray] = {
            axis: Threshold.shrink_axis(X.gram_matrices[axis], threshold[axis], safe=True)
            for axis in X.gram_matrices.keys()
        }
    if self.verbose:
        print("Finished thresholding")

    sep: str = axis_separator

    for axis in X.all_axes:
        if verbose:
            print(f"{axis=}")
        if axis in X.batch_axes:
            continue
        for modality, data in X.dataset.items():
            dimension = X.presences[axis][modality]
            gram_matrix: np.ndarray = threshed_gram[axis]
            thresh: np.ndarray = (gram_matrix + gram_matrix.T) != 0
            if isinstance(thresh, sparse.sparray):
                # igraph does not yet support scipy's sparse array API
                # TODO: Migrate to just using pure scipy as there is a
                # scipy.sparse.csgraph.connected_components function
                # TODO: Allow user defined backends?
                #raise NotImplementedError("igraph does not yet support scipy's sparse array API")
                thresh = sparse.csr_matrix(thresh)
            graph: ig.Graph = ig.Graph.Adjacency(thresh).as_undirected()
            if verbose:
                print("Created Graph")

            # Get connected components
            components = graph.components()
            components_map[axis] = components

            if verbose:
                print(f"Got {len(components)} connected components")
                largest_component_size: int = max(len(component) for component in components)
                print(f"The largest component has size {largest_component_size}")
            for idx, component in enumerate(components):
                if len(component) < min_component_size:
                    to_skip[axis].add(idx)
                    continue

                if isinstance(gram_matrix, sparse.sparray | sparse.spmatrix):
                    new_grams[f'{axis}{sep}{idx}'] = gram_matrix[np.ix_(component, component)]
                else:
                    # coo does not support indexing!
                    new_grams[f'{axis}{sep}{idx}'] = gram_matrix.tocsr()[np.ix_(component, component)].tocoo()

                if isinstance(data, sparse.sparray | sparse.spmatrix):
                    # Recall that sparse arrays don't support batches, so it
                    # is either first or second axis
                    batchless_pres = X.presences_batchless[axis][modality]
                    if batchless_pres == 0:
                        new_dataset[f'{modality}{sep}{idx}'] = data[component, :]
                    elif batchless_pres == 1:
                        new_dataset[f'{modality}{sep}{idx}'] = data[:, component]
                    else:
                        raise Exception("Something interesting went wrong...")
                else:
                    new_dataset[f'{modality}{sep}{idx}'] = data.take(component, axis=dimension)
                new_structure[f'{modality}{sep}{idx}'] = list(X.structure[modality])
                new_structure[f'{modality}{sep}{idx}'][dimension] = f'{axis}||{idx}'
                new_structure[f'{modality}{sep}{idx}'] = tuple(new_structure[f'{modality}||{idx}'])

        X.dataset = new_dataset
        X.structure = new_structure
        X.recompute_helpers()

        new_dataset = dict({})
        new_structure = dict({})

    X.gram_matrices = new_grams

    X.components_map = components_map
    X.to_skip = to_skip
    if verbose:
        print("Final keys:")
        print(f"\t{set(X.gram_matrices.keys())}")
    return X

def recombine_precisions(
    X: Dataset,
    axis_separator: str = "||",
    verbose: bool = False
) -> Dataset:
    """
    Modifies X in place

    TODO: Make this more efficient
    TODO: Check if this still works
    """
    batch_size: int = 1
    new_precision_matrices: dict[Axis, np.ndarray] = {}

    for axis, sz in X.dataset.old_axis_sizes.items():
        new_precision_matrices[axis] = \
            sparse.dok_array((sz, sz), dtype=np.float32)

    for axis, Psi in X.precision_matrices.items():
        if verbose:
            print(axis)
        original_axis: Axis = axis.split(axis_separator)[0]
        group_id: int = int(axis.split(axis_separator)[1])

        if group_id in X.to_skip[original_axis]:
            continue
        original_locations = np.where(
            np.array(X.components_map[original_axis].membership) == group_id
        )[0]
        for idx in range(0, len(original_locations), batch_size):
            if verbose:
                print(idx)
            # Do it row-wise because I think scipy creates a dense matrix
            # when you do bulk indexing
            # (Trying to assign it makes me run out of memory if I do it)
            next_ = min(idx + batch_size, len(original_locations))
            rows = original_locations[idx:next_]
            new_precision_matrices[original_axis][
                np.ix_(rows, original_locations)
            ] = Psi[idx:next_, :]

    X.precision_matrices = new_precision_matrices
    return X