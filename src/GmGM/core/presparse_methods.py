"""
This file contains the methods that are of the following form:
    input -> densifying operation -> sparsifying projection -> output

An example would be a method that takes our eigenvectors and eigenvalues,
creates a precision matrix, and then thresholds that precision matrix.

Rather than ever creating the dense matrix, we can do the operations simultaneously,
preserving sparsity.
"""

from __future__ import annotations
from typing import Literal
import warnings

from ..typing import DataTensor
from ..dataset import Dataset, Axis
import scipy.sparse as sparse
import scipy.stats as stats
import numpy as np
import numba as nb

# ------------------------------------ #
#            Helper Methods            #
# ------------------------------------ #
@nb.jit(nopython=True, fastmath=True, parallel=True, cache=True)
def k_largest_per_row(
    matrix: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Equivalent to
    np.argpartition(
        matrix,
        -k,
        axis=0
    )[-k:]

    But it is much faster. (At least on large matrices)
    """
    output = np.zeros((matrix.shape[0], k), dtype=np.int32)
    for i in nb.prange(matrix.shape[0]):
        row = matrix[i]
        
        to_fill = output[i]
        to_fill[:] = np.arange(k)
        values = row[:k]

        for j in range(k, row.size):
            if row[j] > values[0]:
                to_fill[0] = j
                values[0] = row[j]
                for l in range(1, k):
                    if values[l] > values[l-1]:
                        break
                    else:
                        to_fill[l-1], to_fill[l] = to_fill[l], to_fill[l-1]
                        values[l-1], values[l] = values[l], values[l-1]

    return output



# ------------------------------------ #
#              Public API              #
# ------------------------------------ #
def sparse_grammify(
    X: Dataset,
    to_keep: float | dict[Axis, float],
    threshold_method: Literal["overall", "rowwise", "rowwise-col-weighted"] = "overall",
    gram_method: Literal["covariance", "nonparanormal skeptic"] = "covariance",
    batch_size: int = 100,
    verbose: bool = False
) -> Dataset:
    """
    Creates a pre-thresholded gram matrix,
    meant for use with covariance subdivisions

    Modifies the dataset in-place
    """
    # Clear old gram matrices
    X.gram_matrices = {}

    # Calculate effective gram matrices
    num_samples: dict[Axis, int] = {}
    
    for modality, tensor in X.dataset.items():
        for idx, axis in enumerate(X.structure[modality]):
            if axis in X.batch_axes:
                continue

            matricized: np.ndarray | sparse.sparray
            if isinstance(tensor, sparse.sparray):
                # Handle sparse matrices - note that
                # these are always a matrix, and hence
                # have two axes and no batches.
                # In this case, one axis needs just the tensor
                # and the other needs the transpose of the tensor
                matricized: sparse.sparray = tensor
                if X.presences[axis][modality] == 1:
                    matricized = matricized.T
                matricized = matricized.tocsr()
            else:
                matricized: np.ndarray = np.reshape(
                    np.moveaxis(tensor, idx, 0),
                    (tensor.shape[idx], -1),
                    order='F'
                )

            if verbose:
                print(axis, modality)

            if gram_method == "covariance":
                if threshold_method == "overall":
                    to_add = _estimate_sparse_gram(
                        matricized,
                        to_keep[axis],
                        batch_size,
                        verbose=verbose
                    )
                elif threshold_method == "rowwise":
                    to_add = _estimate_sparse_gram_per_row(
                        matricized,
                        to_keep[axis],
                        batch_size,
                        verbose=verbose
                    )
                elif threshold_method == "rowwise-col-weighted":
                    to_add = _estimate_sparse_gram_per_row_col_weighted(
                        matricized,
                        to_keep[axis],
                        batch_size,
                        verbose=verbose
                    )
            elif gram_method == "nonparanormal skeptic":
                if threshold_method == "overall":
                    to_add = _estimate_sparse_nonparanormal_skeptic(
                        matricized,
                        to_keep[axis],
                        batch_size,
                        verbose=verbose
                    )
                else:
                    raise NotImplementedError("Nonparanormal Skeptic only supports overall thresholding")
            else:
                raise ValueError(
                    f"Unknown method {self.gram_method}"
                    + "\nPlease choose either 'covariance' or 'nonparanormal skeptic'"
                )
            new_samples = X.full_sizes[modality] // X.axis_sizes[axis]
            if axis in X.gram_matrices:
                X.gram_matrices[axis] += to_add
                num_samples[axis] += new_samples
            else:
                X.gram_matrices[axis] = to_add
                num_samples[axis] = new_samples

    # Divide gram matrices by effective number of modalities of samples seen
    for axis in X.gram_matrices.keys():
        X.gram_matrices[axis] /= num_samples[axis]

    # TODO: Remove this; but while I depend on igraph I cannot use sparse arrays
    # but rather I need sparse matrices
    X.gram_matrices = {
        axis: sparse.csr_array(mat)
        for axis, mat in X.gram_matrices.items()
    }

    return X

def recompose_sparse_precisions(
    X: Dataset,
    to_keep: float | int | dict[Axis, float | int],
    threshold_method: Literal["overall", "rowwise", "rowwise-col-weighted"] = "overall",
    batch_size: int = 100,
    verbose: bool = False
) -> Dataset:
    """
    Creates a pre-thresholded precision matrix
    """

    if isinstance(to_keep, float) | isinstance(to_keep, int):
        to_keep = {
            axis: to_keep
            for axis in X.all_axes
        }

    # Calculate the precision matrices
    X.precision_matrices: dict[Axis, np.ndarray] = {}
    for axis in X.all_axes:
        if verbose:
            print(f"Recomposing {axis}")

        # Ensure no negative eigenvalues,
        # necessary to allow square rooting
        negatives = X.evals[axis] < 0
        if negatives.any():
            warnings.warn(
                f"Found negative eigenvalues in axis {axis}"
                + "\nSetting them to 0"
            )
        X.evals[axis][negatives] = 0

        # half @ half.T will equal our precision matrix
        half = X.evecs[axis] * np.sqrt(X.evals[axis])

        # Simultaneously compute and threshold our precision matrix
        if threshold_method == "overall":
            X.precision_matrices[axis] = _estimate_sparse_gram(
                half,
                to_keep[axis],
                batch_size,
                verbose=verbose
            )
        elif threshold_method == "rowwise":
            X.precision_matrices[axis] = _estimate_sparse_gram_per_row(
                half,
                to_keep[axis],
                batch_size,
                verbose=verbose
            )
        elif threshold_method == "rowwise-col-weighted":
            X.precision_matrices[axis] = _estimate_sparse_gram_per_row_col_weighted(
                half,
                to_keep[axis],
                batch_size,
                verbose=verbose
            )
        else:
            raise ValueError(f"{threshold_method} is not a valid thresholding method")
        
    return X

# ------------------------------------ #
#            Helper Methods            #
# ------------------------------------ #
def _estimate_sparse_gram(
    matrix: DataTensor,
    percent_to_keep: float,
    batch_size: int = 1,
    divide_by_diagonal: bool = False,
    verbose: bool = False
) -> sparse.sparray:
    """
    Computes matrix @ matrix.T in a sparse manner.
    """

    assert percent_to_keep >= 0 and percent_to_keep <= 1, \
        "`percent_to_keep` must be between 0 and 1"

    features, _ = matrix.shape
    num_to_keep: int = int(percent_to_keep * (features**2 - features) / 2)

    # Create a sparse array to store the result
    # We are always gonna keep it so smallest kept value is at `num_to_keep`
    # Note that this uses twice as much memory as we ultimately will need,
    # because we want to zipper-sort the old data and new data in the end.
    # If there is a vectorized way to zipper-sort, rather than concatenating them,
    # then we can save this memory.  But idk how!
    result = sparse.coo_array(
        (
            np.zeros(2*num_to_keep, dtype=np.float32),
            (
                np.zeros(2*num_to_keep, dtype=np.int32),
                np.zeros(2*num_to_keep, dtype=np.int32)
            )
        ),
        shape=(features, features),
    )


    for i in range(0, features + batch_size, batch_size):
        if i >= features:
            break
        # Compute the entire row at once to save time
        # Assuming we have space that is linear in axis length
        # (a reasonable assumption)
        # then this is okay to do!
        increase: int = min(batch_size, features - i)
        res_batch: np.ndarray
        if isinstance(matrix, sparse.sparray):
            # Warning: this implementation becomes quite slow
            # when the matrices are very large.  Better to,
            # if possible, work with dense matrices.
            rhs = matrix[i:i+increase].toarray().T
            res_batch = np.abs(matrix[i:] @ rhs)
        else:
            res_batch = np.abs(matrix[i:] @ matrix[i:i+increase].T)

        # Remove the diagonal so it does not affect thresholding!
        diags = np.diagonal(res_batch).copy()
        if divide_by_diagonal:
            diags[diags == 0] = 1
            np.fill_diagonal(res_batch, 0)
        else:
            diags[:] = 1

        # Remove everything above diagonal so it does not affect thresholding!
        # (due to symmetry)
        res_batch[np.triu_indices(increase)] = 0

        # We only need to check the `num_to_keep` largest elements
        # in this batch!
        to_check = min(num_to_keep, res_batch.size)
        flat_res_batch = res_batch.reshape(-1)
        largest_idxs = np.argpartition(
            (res_batch / diags).reshape(-1),
            -to_check,
            axis=None
        )[-to_check:]
        rows = largest_idxs // res_batch.shape[1]
        cols = largest_idxs % res_batch.shape[1]

        # Remove any elements that are smaller than the smallest kept
        candidates = flat_res_batch[largest_idxs] > result.data[num_to_keep]
        flat_res_batch = flat_res_batch[largest_idxs][candidates]
        rows = rows[candidates]
        cols = cols[candidates]

        if flat_res_batch.size == 0:
            # Nothing to add!
            continue

        # Figure out how many elements we can add
        sorted = np.argsort(flat_res_batch)
        flat_res_batch = flat_res_batch[sorted]
        rows = rows[sorted]
        cols = cols[sorted]
        amount = (result.data < flat_res_batch[-1]).sum()
        amount = min(flat_res_batch.size, amount)

        if amount == 0:
            # Nothing to add!
            continue

        # Add these to the data!
        result.data[:amount] = flat_res_batch[-amount:]
        result.row[:amount] = rows[-amount:] + i
        result.col[:amount] = cols[-amount:] + i

        # And sort the data
        # Note that this makes the complexity klogk where k is the number
        # of elements to keep; this is slower than the older implementation
        # using dictionaries and dok_arrays.
        # However, because everything is vectorized here, and k is typically
        # quite small, and log k is smaller still, this is MUCH faster in practice.
        sorted = np.argsort(result.data)
        result.data = result.data[sorted]
        result.row = result.row[sorted]
        result.col = result.col[sorted]

        # Bottleneck (>75-80% runtime) for large (10,000x10,000) data is the matrix multiplication
        # when keeping on average 10 edges per row (i.e. k=100,000).
        # The sorts account for ~5-10% of runtime.
        #
        # Argpartition is linear time in the number of elements to partition, and it takes ~15%.

    # Remove the excess memory
    result.data = result.data[num_to_keep:]
    result.row = result.row[num_to_keep:]
    result.col = result.col[num_to_keep:]

    return result

def _estimate_sparse_gram_per_row(
    matrix: DataTensor,
    per_row_to_keep: float,
    batch_size: int = 1,
    verbose: bool = False
) -> sparse.sparray:
    """
    Computes matrix @ matrix.T in a sparse manner.
    """

    assert per_row_to_keep >= 0 and per_row_to_keep <= matrix.shape[0], \
        "`per_row_to_keep` must be between 0 and the number of features"

    features, _ = matrix.shape
    num_to_keep: int = per_row_to_keep * features

    # Create a sparse array to store the result
    result = sparse.coo_array(
        (
            np.zeros(num_to_keep, dtype=np.float32),
            (
                np.zeros(num_to_keep, dtype=np.int32),
                np.zeros(num_to_keep, dtype=np.int32)
            )
        ),
        shape=(features, features),
    )


    for i in range(0, features + batch_size, batch_size):
        if i >= features:
            break
        # Compute the entire row at once to save time
        # Assuming we have space that is linear in axis length
        # (a reasonable assumption)
        # then this is okay to do!
        increase: int = min(batch_size, features - i)
        res_batch: np.ndarray
        if isinstance(matrix, sparse.sparray):
            # Warning: this implementation becomes quite slow
            # when the matrices are very large.  Better to,
            # if possible, work with dense matrices.
            rhs = matrix[i:i+increase].toarray().T
            res_batch = np.abs(matrix @ rhs)
        else:
            res_batch = np.abs(matrix @ matrix[i:i+increase].T)

        # Remove the diagonal so it does not affect selection!
        np.fill_diagonal(res_batch[i:], 0)

        # We only need to check the `num_to_keep` largest elements
        # in this batch!
        largest_idxs = k_largest_per_row(
            res_batch.T,
            per_row_to_keep,
        ).T
        # largest_idxs = np.argpartition(
        #     res_batch,
        #     -per_row_to_keep,
        #     axis=0
        # )[-per_row_to_keep:]
        rows = largest_idxs.T.reshape(-1)

        # Add these to the data!
        start = i * per_row_to_keep
        end = (i + increase) * per_row_to_keep
        result.data[start:end] = np.take_along_axis(
            res_batch,
            largest_idxs,
            axis=0
        ).T.reshape(-1)
        result.row[start:end] = rows

        for b in range(increase):
            start = (i + b) * per_row_to_keep
            end = (i + 1 + b) * per_row_to_keep
            result.col[start:end] = i + b

    # Remove upper triangle
    result.data[result.row < result.col] = 0

    return result

def _estimate_sparse_gram_per_row_col_weighted(
    matrix: DataTensor,
    per_row_to_keep: float,
    batch_size: int = 1,
    verbose: bool = False
) -> sparse.sparray:
    """
    Computes matrix @ matrix.T in a sparse manner.
    """

    assert per_row_to_keep >= 0 and per_row_to_keep <= matrix.shape[0], \
        "`per_row_to_keep` must be between 0 and the number of features"

    features, _ = matrix.shape
    num_to_keep: int = per_row_to_keep * features

    # Construct column weights - because we are using absolute value,
    # this is more complicated!
    # Equivalent to:
    #   col_weights = np.abs(matrix @ matrix.T).sum(axis=0)
    col_weights = np.zeros(features, dtype=np.float32)
    for i in range(0, features + batch_size, batch_size):
        if i >= features:
            break
        # Compute the entire row at once to save time
        # Assuming we have space that is linear in axis length
        # (a reasonable assumption)
        # then this is okay to do!
        increase: int = min(batch_size, features - i)
        res_batch: np.ndarray
        if isinstance(matrix, sparse.sparray):
            # Warning: this implementation becomes quite slow
            # when the matrices are very large.  Better to,
            # if possible, work with dense matrices.
            rhs = matrix[i:i+increase].toarray().T
            res_batch = np.abs(matrix @ rhs)
        else:
            res_batch = np.abs(matrix @ matrix[i:i+increase].T)

        # Remove the diagonal so it does not affect the sum!
        np.fill_diagonal(res_batch[i:], 0)

        col_weights[i:i+increase] += res_batch.sum(axis=0)


    col_weights[col_weights == 0] = 1
    col_weights = col_weights.reshape(-1, 1)

    # Create a sparse array to store the result
    result = sparse.coo_array(
        (
            np.zeros(num_to_keep, dtype=np.float32),
            (
                np.zeros(num_to_keep, dtype=np.int32),
                np.zeros(num_to_keep, dtype=np.int32)
            )
        ),
        shape=(features, features),
    )


    for i in range(0, features + batch_size, batch_size):
        if i >= features:
            break
        # Compute the entire row at once to save time
        # Assuming we have space that is linear in axis length
        # (a reasonable assumption)
        # then this is okay to do!
        increase: int = min(batch_size, features - i)
        res_batch: np.ndarray
        if isinstance(matrix, sparse.sparray):
            # Warning: this implementation becomes quite slow
            # when the matrices are very large.  Better to,
            # if possible, work with dense matrices.
            rhs = matrix[i:i+increase].toarray().T
            res_batch = np.abs(matrix @ rhs)
        else:
            res_batch = np.abs(matrix @ matrix[i:i+increase].T)

        # Remove the diagonal so it does not affect selection!
        np.fill_diagonal(res_batch[i:], 0)

        # We only need to check the `per_row_to_keep` largest elements
        # in this batch!
        largest_idxs = k_largest_per_row(
            (res_batch / col_weights).T,
            per_row_to_keep,
        ).T
        rows = largest_idxs.T.reshape(-1)

        # print(res_batch.shape)
        # print(per_row_to_keep)
        # print((res_batch / col_weights)[largest_idxs])
        # print((res_batch / col_weights).max(axis=0))
        # print("---")

        # Add these to the data!
        start = i * per_row_to_keep
        end = (i + increase) * per_row_to_keep
        result.data[start:end] = np.take_along_axis(
            res_batch,
            largest_idxs,
            axis=0
        ).T.reshape(-1)
        result.row[start:end] = rows

        for b in range(increase):
            start = (i + b) * per_row_to_keep
            end = (i + 1 + b) * per_row_to_keep
            result.col[start:end] = i + b

    # Remove upper triangle
    #result.data[result.row < result.col] = 0

    # #print(result.data)
    # print(result.row)
    # print(result.col)

    return result

def _estimate_sparse_nonparanormal_skeptic(
    matrix: DataTensor,
    percent_to_keep: float,
    batch_size: int = 1,
    divide_by_diagonal: bool = True,
    verbose: bool = False
) -> sparse.sparray:
    """
    Computes matrix @ matrix.T in a sparse manner.
    """

    assert percent_to_keep >= 0 and percent_to_keep <= 1, \
        "`percent_to_keep` must be between 0 and 1"

    features, _ = matrix.shape
    num_to_keep: int = int(percent_to_keep * (features**2 - features) / 2)

    # Create a sparse array to store the result
    # We are always gonna keep it so smallest kept value is at `num_to_keep`
    # Note that this uses twice as much memory as we ultimately will need,
    # because we want to zipper-sort the old data and new data in the end.
    # If there is a vectorized way to zipper-sort, rather than concatenating them,
    # then we can save this memory.  But idk how!
    result = sparse.coo_array(
        (
            np.zeros(2*num_to_keep, dtype=np.float32),
            (
                np.zeros(2*num_to_keep, dtype=np.int32),
                np.zeros(2*num_to_keep, dtype=np.int32)
            )
        ),
        shape=(features, features),
    )

    # Replace matrix with ranks
    matrix = stats.rankdata(matrix, axis=1, nan_policy='raise')
    matrix = matrix - matrix.mean(axis=1, keepdims=True)

    # Get square root of diagonals of matrix
    sqrt_diags = np.zeros(features, dtype=np.float32)
    for i in range(0, features):
        sqrt_diags[i] = 1/np.sqrt(matrix[i] @ matrix[i])

    for i in range(0, features + batch_size, batch_size):
        if i >= features:
            break
        # Compute the entire row at once to save time
        # Assuming we have space that is linear in axis length
        # (a reasonable assumption)
        # then this is okay to do!
        increase: int = min(batch_size, features - i)
        res_batch: np.ndarray

        # Calculate correlation coefficient of matrix of ranks
        res_batch = matrix[i:] @ matrix[i:i+increase].T
        res_batch = (
            sqrt_diags[i:i+increase].reshape(1, -1)
            * res_batch
            * sqrt_diags[i:].reshape(-1, 1)
        )
        res_batch = np.abs(2 * np.sin(np.pi / 6 * res_batch))

        # Remove the diagonal so it does not affect thresholding!
        np.fill_diagonal(res_batch, 0)

        # Remove everything above diagonal so it does not affect thresholding!
        # (due to symmetry)
        res_batch[np.triu_indices(increase)] = 0

        # We only need to check the `num_to_keep` largest elements
        # in this batch!
        to_check = min(num_to_keep, res_batch.size)
        flat_res_batch = res_batch.reshape(-1)
        largest_idxs = np.argpartition(
            flat_res_batch,
            -to_check,
            axis=None
        )[-to_check:]
        rows = largest_idxs // res_batch.shape[1]
        cols = largest_idxs % res_batch.shape[1]

        # Remove any elements that are smaller than the smallest kept
        candidates = flat_res_batch[largest_idxs] > result.data[num_to_keep]
        flat_res_batch = flat_res_batch[largest_idxs][candidates]
        rows = rows[candidates]
        cols = cols[candidates]

        if flat_res_batch.size == 0:
            # Nothing to add!
            continue

        # Figure out how many elements we can add
        sorted = np.argsort(flat_res_batch)
        flat_res_batch = flat_res_batch[sorted]
        rows = rows[sorted]
        cols = cols[sorted]
        amount = (result.data < flat_res_batch[-1]).sum()
        amount = min(flat_res_batch.size, amount)

        if amount == 0:
            # Nothing to add!
            continue

        # Add these to the data!
        result.data[:amount] = flat_res_batch[-amount:]
        result.row[:amount] = rows[-amount:] + i
        result.col[:amount] = cols[-amount:] + i

        # And sort the data
        # Note that this makes the complexity klogk where k is the number
        # of elements to keep; this is slower than the older implementation
        # using dictionaries and dok_arrays.
        # However, because everything is vectorized here, and k is typically
        # quite small, and log k is smaller still, this is MUCH faster in practice.
        sorted = np.argsort(result.data)
        result.data = result.data[sorted]
        result.row = result.row[sorted]
        result.col = result.col[sorted]

        # Bottleneck (>75-80% runtime) for large (10,000x10,000) data is the matrix multiplication
        # when keeping on average 10 edges per row (i.e. k=100,000).
        # The sorts account for ~5-10% of runtime.
        #
        # Argpartition is linear time in the number of elements to partition, and it takes ~15%.

    # Remove the excess memory
    result.data = result.data[num_to_keep:]
    result.row = result.row[num_to_keep:]
    result.col = result.col[num_to_keep:]

    return result