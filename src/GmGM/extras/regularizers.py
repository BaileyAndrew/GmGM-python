from __future__ import annotations

import numpy as np
from ..typing import MaybeDict, Axis
from numbers import Number

class Regularizer:
    """
    A subclassable regularizer class.
    
    Has a method to get the gradient the regularizer
        on a set of eigenvalues and eigenvectors.
        and a method to get the loss of the regularizer.

    """

    def __init__(
        self,
        rhos: MaybeDict[Axis, np.ndarray],
        compute_loss: bool = False,
    ):
        """
        Initializes the regularizer with a set of penalties `rhos`.

        If compute_loss is false, does not compute loss terms each iteration
        """
        self.rhos = rhos
        self.compute_loss = compute_loss

    def loss(
        self,
        evals: dict[Axis, np.ndarray],
        evecs: dict[Axis, np.ndarray],
    ) -> float:
        """
        Returns the loss of the regularizer
            on a set of eigenvalues and eigenvectors.
        """
        pass

    def prox(
        self,
        evals: dict[Axis, np.ndarray],
        evecs: dict[Axis, np.ndarray],
    ) -> dict[Axis, np.ndarray]:
        """
        The proximal operator of the regularizer.
        """
        pass

class SpectralAbsoluteL1(Regularizer):
    """
    L1 regularizer of off-diagonal elements of the precision matrix, where the evecs are known
        and the evals are all that vary.

    Calculating the loss is quite expensive so I'd recommend compute_loss to be false

    TODO: update docstring, this does not quite equal the Sectral L1 regularizer
    """

    def __init__(
        self,
        rhos: MaybeDict[Axis, np.ndarray],
        compute_loss: bool = False,
    ):
        """
        Initializes the regularizer with a set of penalties `rhos`.

        If compute_loss is false, does not compute loss terms each iteration
        """
        super().__init__(rhos, compute_loss)

        self.shift_terms = {}

    def loss(
        self,
        evals: dict[Axis, np.ndarray],
        evecs: dict[Axis, np.ndarray],
    ) -> float:
        if not self.compute_loss:
            return 0
        to_return = 0
        for axis in evals.keys():
            reconstructed = (evecs[axis] * evals[axis]) @ evecs[axis].T
            to_return += self.rhos[axis] * np.abs(reconstructed).sum()
        return to_return
    
    def prox(
        self,
        evals: dict[Axis, np.ndarray],
        evecs: dict[Axis, np.ndarray],
        t: float
    ) -> dict[Axis, np.ndarray]:
        to_return = {}
        for axis in evals.keys():
            eval = evals[axis]
            evec = evecs[axis]
            if isinstance(self.rhos, Number):
                rho = t * self.rhos
            else:
                rho = t * self.rhos[axis]

            if axis not in self.shift_terms:
                # Calculate for the first time
                abs_evec = np.abs(evec)
                shift_term = np.einsum("ai, bi -> i", abs_evec, abs_evec)
                self.shift_terms[axis] = shift_term
            else:
                # Stored computed shift term
                shift_term = self.shift_terms[axis]

            if np.isinf(rho):
                # As rho gets arbitrarily large, the proximal operator converges to
                # the shift_term.
                to_return[axis] = shift_term
                continue

            # This is the proximal operator
            to_return[axis] = (eval + rho * shift_term) / (1 + rho)

        return to_return