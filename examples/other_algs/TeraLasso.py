"""
This is a python wrapper for TeraLasso
"""

import matlab.engine
import io
import numpy as np
from GmGM import Dataset
from GmGM.core.preprocessing import create_gram_matrices
from GmGM.typing import Axis
import scipy.sparse as sparse

eng = matlab.engine.start_matlab()
eng.addpath(
    './teralasso'
)


def TeraLasso(
    dataset: Dataset,
    beta: float,
    use_nonparanormal_skeptic: bool = False,
    max_iter: int = 100,
    tol: float = 1e-8
) -> Dataset:
    if len(dataset.dataset) != 1:
        raise ValueError(
            'TeraLasso only supports one dataset'
        )
    if len(dataset.batch_axes) == 0:
        raise ValueError(
            "Please make the first axis of the dataset a batch axis!"
        )
    tensor = list(dataset.dataset.values())[0]
    structure = list(dataset.structure.values())[0]
    if structure[0] != '':
        tensor = tensor.reshape(1, *tensor.shape)
    _, *d = tensor.shape
    K = len(d)
    
    dataset = create_gram_matrices(
        dataset,
        use_nonparanormal_skeptic=use_nonparanormal_skeptic,
    )
    Ss = {
        axis: matlab.double(matrix)
        for axis, matrix
        in dataset.gram_matrices.items()
    }

    d_matlab = matlab.double(d)
    betas_matlab = matlab.double([beta for _ in range(K)])
    tol_matlab = matlab.double(tol)

    Psis_matlab = eng.teralasso(
        [matrix for _, matrix in Ss.items()],
        d_matlab,
        'L1',
        0,
        tol_matlab,
        betas_matlab,
        max_iter,
        nargout=1,
        stdout=io.StringIO()
    )
    
    Psis = {}
    
    for (axis, _), Psi in zip(Ss.items(), Psis_matlab):
        Psis[axis] = sparse.csr_array(np.asarray(Psi))
    
    dataset.precision_matrices = Psis
    return dataset