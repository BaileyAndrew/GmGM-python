"""
Some bits of code were originally written in Fortran,
however I have now ported them to Python+Numba as:
1) Pure python code is easier to distribute
2) Numba actually turned out to be faster!

I think the reason why is that I do not know much about
compilers so I did not choose the best settings to enable
parallelism, but Numba makes it quite easy.
"""

import numba as nb
import numpy as np
from .typing import Axis, Modality

class _project_inv_kron_sum:
    @staticmethod
    @nb.jit(nopython=True, fastmath=True, parallel=True, cache=True)
    def two_axis(
        x: np.ndarray,
        y: np.ndarray
    ) -> tuple[np.ndarray]:
        
        k_ratio: float = 1./2.
        x_out: np.ndarray = np.zeros(x.shape[0])
        y_out: np.ndarray = np.zeros(y.shape[0])

        for i in nb.prange(x.shape[0]):
            for j in nb.prange(y.shape[0]):
                cur_val: float = 1 / (x[i]+y[j])
                x_out[i] += cur_val
                y_out[j] += cur_val

        # Normalize
        x_out /= x.shape[0]
        y_out /= y.shape[0]

        # Offset diagonal
        x_out -= k_ratio * np.sum(x_out) / x.shape[0]
        y_out -= k_ratio * np.sum(y_out) / y.shape[0]

        return x_out, y_out

    @staticmethod
    @nb.jit(nopython=True, fastmath=True, parallel=True, cache=True)
    def three_axis(
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray
    ) -> tuple[np.ndarray]:
        
        k_ratio: float = 2./3.
        x_out: np.ndarray = np.zeros(x.shape[0])
        y_out: np.ndarray = np.zeros(y.shape[0])
        z_out: np.ndarray = np.zeros(z.shape[0])

        for i in nb.prange(x.shape[0]):
            for j in nb.prange(y.shape[0]):
                for k in nb.prange(z.shape[0]):
                    cur_val: float = 1 / (x[i]+y[j]+z[k])
                    x_out[i] += cur_val
                    y_out[j] += cur_val
                    z_out[j] += cur_val

        # Normalize
        x_out /= x.shape[0]
        y_out /= y.shape[0]
        z_out /= z.shape[0]

        # Offset diagonal
        x_out -= k_ratio * np.sum(x_out) / x.shape[0]
        y_out -= k_ratio * np.sum(y_out) / y.shape[0]
        z_out -= k_ratio * np.sum(z_out) / z.shape[0]

        return x_out, y_out, z_out
    
    @nb.jit(nopython=True, fastmath=True, parallel=True, cache=True)
    def four_axis(
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        w: np.ndarray
    ) -> tuple[np.ndarray]:
        
        k_ratio: float = 3./4.
        x_out: np.ndarray = np.zeros(x.shape[0])
        y_out: np.ndarray = np.zeros(y.shape[0])
        z_out: np.ndarray = np.zeros(z.shape[0])
        w_out: np.ndarray = np.zeros(w.shape[0])

        for i in nb.prange(x.shape[0]):
            for j in nb.prange(y.shape[0]):
                for k in nb.prange(z.shape[0]):
                    for l in nb.prange(w.shape[0]):
                        cur_val: float = 1 / (x[i]+y[j]+z[k]+w[l])
                        x_out[i] += cur_val
                        y_out[j] += cur_val
                        z_out[j] += cur_val
                        w_out[j] += cur_val

        # Normalize
        x_out /= x.shape[0]
        y_out /= y.shape[0]
        z_out /= z.shape[0]
        w_out /= w.shape[0]

        # Offset diagonal
        x_out -= k_ratio * np.sum(x_out) / x.shape[0]
        y_out -= k_ratio * np.sum(y_out) / y.shape[0]
        z_out -= k_ratio * np.sum(z_out) / z.shape[0]
        w_out -= k_ratio * np.sum(w_out) / w.shape[0]

        return x_out, y_out, z_out, w_out
    
class _sum_log_sum:
    @staticmethod
    @nb.jit(nopython=True, fastmath=True, parallel=True, cache=True)
    def two_axis(
        x: np.ndarray,
        y: np.ndarray,
        simplify_size: int = 20
    ) -> float:
        
        sl: float = 0
        intermediate: float = 1
        
        for i in nb.prange(x.shape[0]):
            for j in nb.prange(y.shape[0]):
                sm: float = x[i] + y[j]
                intermediate *= sm

                if (i+j) % simplify_size == 0:
                    sl += np.log(intermediate)
                    intermediate = 1.0

        sl += np.log(intermediate)

        return sl
    
    @staticmethod
    @nb.jit(nopython=True, fastmath=True, parallel=True, cache=True)
    def three_axis(
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        simplify_size: int = 20
    ) -> float:
        
        sl: float = 0
        intermediate: float = 1
        
        for i in nb.prange(x.shape[0]):
            for j in nb.prange(y.shape[0]):
                for k in nb.prange(z.shape[0]):
                    sm: float = x[i] + y[j] + z[k]
                    intermediate *= sm

                    if (i+j+k) % simplify_size == 0:
                        sl += np.log(intermediate)
                        intermediate = 1.0

        sl += np.log(intermediate)

        return sl
    
    @staticmethod
    @nb.jit(nopython=True, fastmath=True, parallel=True, cache=True)
    def four_axis(
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        w: np.ndarray,
        simplify_size: int = 20
    ) -> float:
        
        sl: float = 0
        intermediate: float = 1
        
        for i in nb.prange(x.shape[0]):
            for j in nb.prange(y.shape[0]):
                for k in nb.prange(z.shape[0]):
                    for l in nb.prange(w.shape[0]):
                        sm: float = x[i] + y[j] + z[k] + w[l]
                        intermediate *= sm

                        if (i+j+k+l) % simplify_size == 0:
                            sl += np.log(intermediate)
                            intermediate = 1.0

        sl += np.log(intermediate)

        return sl

_proj_iks: dict[int, callable] = {
    2: _project_inv_kron_sum.two_axis,
    3: _project_inv_kron_sum.three_axis,
    4: _project_inv_kron_sum.four_axis,
}
_sls: dict[int, callable] = {
    2: _sum_log_sum.two_axis,
    3: _sum_log_sum.three_axis,
    4: _sum_log_sum.four_axis,
}
          

def project_inv_kron_sum(
    evals: dict[Axis, np.ndarray],
    structure: dict[Modality, tuple[Axis]],
    modality: Modality,
    batch_axes: set[Axis],
    K: int
):
    args = [
        evals[axis]
        for axis in structure[modality]
        if axis not in batch_axes
    ]
    
    if K not in _proj_iks:
        raise ValueError(
            "Only 2-4 way tensors are supported!"
            + f"... What are you doing with {K} axes?"
        )
    return _proj_iks[K](*args)

def sum_log_sum(
        *args: list[np.array]
    ) -> float:
        """
        Computes:
            the sum
            of the log
            of the determinant
            of the kronecker sum
            of the input matrices 
        """
        K = len(args)
        if K not in _sls:
            raise ValueError(
                "Only 2-4 way tensors are supported!"
                + f"... What are you doing with {K} axes?"
            )
        return _sls[K](*args)

def extract_d_values(shape):
    d_all = np.prod(shape)
    d_lefts = np.cumprod((1, *shape))[:-1]
    d_rights = np.cumprod((1, *shape[::-1]))[::-1][1:]

    return d_all, d_lefts, d_rights