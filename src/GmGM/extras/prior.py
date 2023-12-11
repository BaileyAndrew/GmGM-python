"""
Provides the Prior class
"""

from __future__ import annotations

from typing import Literal
import numpy as np

class Prior:
    def __init__(self) -> None:
        pass

class Wishart(Prior):
    name = "Wishart"
    def __init__(
        self,
        *,
        degrees_of_freedom: int,
        scale_matrix: np.ndarray
    ):
        self.degrees_of_freedom = degrees_of_freedom
        self.scale_matrix = scale_matrix

        p: int = self.scale_matrix.shape[0]

        assert self.scale_matrix.shape == (p, p), \
            "Scale matrix must be square"
        
        assert degrees_of_freedom > p - 1, \
            "Degrees of freedom must be greater than p - 1"
        
        self.subtract_from_gram = -np.linalg.inv(self.scale_matrix)

    def process_gram(
        self,
        gram: np.ndarray
    ) -> np.ndarray:
        # Eta = -1/2 * prior^-1
        # Drop the 1/2 as we ignore it everywhere
        # Subtract from Gram as in Theorem 3
        return gram - self.subtract_from_gram
    
    def process_gradient(
        self,
        evals: np.ndarray
    ) -> np.ndarray:
        # h = -1/2 * precmat^-1
        # drop the 1/2 as we ignore it everywhere
        return -1/evals