"""
This is a python wrapper for EiGLasso
"""

import matlab.engine
import io
import numpy as np
from GmGM import Dataset
from GmGM.core.preprocessing import create_gram_matrices
from GmGM.typing import Axis
import warnings

eng = matlab.engine.start_matlab()

eng.addpath(
    './EiGLasso/EiGLasso_JMLR'
)

    
def EiGLasso(
    dataset: Dataset,
    beta_1: float,
    beta_2: float,
    use_nonparanormal_skeptic: bool = False,
    verbose: bool = False
) -> Dataset:
    if len(dataset.dataset) != 1:
        raise ValueError(
            'EiGLasso only supports one dataset'
        )
    tensor = list(dataset.dataset.values())[0]
    _, *d = tensor.shape
    K = len(d)
    if K != 2:
        raise ValueError(
            'EiGLasso only supports 2 dimensions'
        )
    
    dataset = create_gram_matrices(
        dataset,
        use_nonparanormal_skeptic=use_nonparanormal_skeptic,
    )

    Ss = {
        axis: matrix
        for axis, matrix
        in dataset.gram_matrices.items()
    }

    # Convert to matlab format
    S = list(Ss.values())[0]
    T = list(Ss.values())[1]
    T_ = matlab.double(T)
    S_ = matlab.double(S)
    beta_1 = matlab.double(beta_1)
    beta_2 = matlab.double(beta_2)


    stdout = {} if verbose else {"stdout": io.StringIO()}
    
    Psi_init = np.linalg.inv(T)
    Psi_init = matlab.double(Psi_init)
    Theta_init = np.linalg.inv(S)
    Theta_init = matlab.double(Theta_init)
    
    # Call matlab (which itself calls the
    # compiled c++ file `eiglasso_joint_mex.cpp`)
    Theta, Psi, _, _ = eng.eiglasso_joint(
        S_,
        T_,
        beta_2,
        beta_1,
        nargout=4,
        **stdout
    )
    
    # Convert to python format
    Theta = np.asarray(Theta)
    Psi = np.asarray(Psi)
    
    # They're not symmetric though...
    Theta = (Theta + Theta.T) / 2
    Psi = (Psi + Psi.T) / 2
    
    dataset.precision_matrices = {
        axis: matrix
        for axis, matrix
        in zip(Ss.keys(), [Theta, Psi])
    }

    return dataset