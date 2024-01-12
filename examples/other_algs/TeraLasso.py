"""
This is a python wrapper for TeraLasso
"""

import matlab.engine
import io
import numpy as np
from GmGM import Dataset
from GmGM.core.preprocessing import create_gram_matrices
from GmGM.typing import Axis

eng = matlab.engine.start_matlab()
eng.addpath(
    './teralasso'
)


def TeraLasso(
    dataset: Dataset,
    beta: float,
    use_nonparanormal_skeptic: bool = False,
    max_iter: int = 100
) -> Dataset:
    if len(dataset.dataset) != 1:
        raise ValueError(
            'TeraLasso only supports one dataset'
        )
    tensor = list(dataset.dataset.values())[0]
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

    Psis_matlab = eng.teralasso(
        [matrix for _, matrix in Ss.items()],
        d_matlab,
        'L1',
        0,
        1e-8,
        betas_matlab,
        max_iter,
        nargout=1,
        stdout=io.StringIO()
    )
    
    Psis = {}
    
    for (axis, _), Psi in zip(Ss.items(), Psis_matlab):
        Psis[axis] = np.asarray(Psi)
    
    dataset.precision_matrices = Psis
    return dataset