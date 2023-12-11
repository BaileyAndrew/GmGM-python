from __future__ import annotations

import numpy as np

class Regularizer:
    """
    A subclassable regularizer class.
    
    Has a method to get the gradient the regularizer
        on a set of eigenvalues and eigenvectors.
        and a method to get the loss of the regularizer.

    """

    def __init__(
        self,
        rhos: dict[str, np.ndarray]
    ):
        """
        Initializes the regularizer with a set of penalties `rhos`.
        """
        self.rhos = rhos

    def loss(
        self,
        evals: dict[str, np.ndarray],
        evecs: dict[str, np.ndarray],
    ) -> float:
        """
        Returns the loss of the regularizer
            on a set of eigenvalues and eigenvectors.
        """
        pass

    def grad(
        self,
        evals: dict[str, np.ndarray],
        evecs: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """
        Returns the gradient of the regularizer
            on a set of eigenvalues and eigenvectors.
        """
        pass

class L1(Regularizer):
    """
    Restricted L1 Regularizer.
    """

    def __init__(
        self,
        rhos: dict[str, np.ndarray]
    ):
        super().__init__(rhos)

        # Keep track of reconstructed matrices
        #  to avoid recomputing them.

        self.reconstructions: np.ndarray = {}

    def loss(
        self,
        evals: dict[str, np.ndarray],
        evecs: dict[str, np.ndarray],
    ) -> float:
        to_return = 0
        for axis, recon in self.reconstructions.items():
            to_return += self.rhos[axis] * np.abs(recon).sum()
        return to_return

    def grad(
        self,
        evals: dict[str, np.ndarray],
        evecs: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        to_return: dict[str, np.ndarray] = {}
        for axis in evals.keys():
            if self.rhos[axis] == 0:
                to_return[axis] = np.zeros_like(evals[axis])
                continue
            self.reconstructions[axis] = (
                (evecs[axis] * evals[axis]) @ evecs[axis].T
            )
            np.fill_diagonal(self.reconstructions[axis], 0)
            core: np.ndarray = np.sign(
                self.reconstructions[axis]
            )
            to_return[axis] = self.rhos[axis] * np.einsum(
                "ij, ia, ja -> a",
                core,
                evecs[axis],
                evecs[axis],
                optimize=True
            )
        return to_return
    
class L1Approx(Regularizer):
    """
    L1 Regularizer using an approximate reconstruction
    """

    def __init__(
        self,
        rhos: dict[str, np.ndarray],
        num_eigs: int = 1
    ):
        super().__init__(rhos)

        self.num_eigs = num_eigs

        # Keep track of reconstructed matrices
        #  to avoid recomputing them.
        self.reconstructions: np.ndarray = {}

    def loss(
        self,
        evals: dict[str, np.ndarray],
        evecs: dict[str, np.ndarray],
    ) -> float:
        to_return = 0
        for axis, recon in self.reconstructions.items():
            to_return += self.rhos[axis] * np.abs(recon).sum()
        return to_return

    def grad(
        self,
        evals: dict[str, np.ndarray],
        evecs: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        to_return: dict[str, np.ndarray] = {}
        for axis in evals.keys():
            if self.rhos[axis] == 0:
                to_return[axis] = np.zeros_like(evals[axis])
                continue
            # Grab the top `num_eigs` eigenvalues and eigenvectors
            #  and reconstruct the matrix from them.
            top_idx = np.argpartition(evals[axis], -self.num_eigs)[-self.num_eigs:]

            top_evecs = evecs[axis][:, top_idx]
            top_evals = evals[axis][top_idx]

            self.reconstructions[axis] = (
                (top_evecs * top_evals) @ top_evecs.T
            )

            np.fill_diagonal(self.reconstructions[axis], 0)
            core: np.ndarray = np.sign(
                self.reconstructions[axis]
            )
            to_return[axis] = np.zeros_like(evals[axis])
            to_return[axis][top_idx] = self.rhos[axis] * np.einsum(
                "ij, ia, ja -> a",
                core,
                top_evecs,
                top_evecs,
                optimize=True
            )

        return to_return
    
class L2(Regularizer):
    """
    L2 Regularizer

    This is same as the Frobenius norm.
    Hence, we can just take the magnitude of
        the eigenvalues!
    """

    def __init__(
        self,
        rhos: dict[str, np.ndarray]
    ):
        super().__init__(rhos)

        # Keep track of magnitudes to avoid recomputations
        self.magnitudes: dict[str, float] = {}
    

    def loss(
        self,
        evals: dict[str, np.ndarray],
        evecs: dict[str, np.ndarray],
    ) -> float:
        to_return = 0
        for axis in evals.keys():
            if self.rhos[axis] == 0:
                continue
            to_return += self.rhos[axis] * self.magnitudes[axis]
        return to_return
    
    def grad(
        self,
        evals: dict[str, np.ndarray],
        evecs: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        to_return: dict[str, np.ndarray] = {}
        for axis in evals.keys():
            if self.rhos[axis] == 0:
                to_return[axis] = np.zeros_like(evals[axis])
                continue
            self.magnitudes[axis] = np.linalg.norm(evals[axis])
            to_return[axis] = self.rhos[axis] / self.magnitudes[axis] * evals[axis]
        return to_return
    
class Random(Regularizer):
    """
    Not actually a regularizer, just adds randomness to the gradient.

    Does poorly.
    """

    def __init__(
        self,
        rhos: dict[str, np.ndarray]
    ):
        super().__init__(rhos)

    def loss(
        self,
        evals: dict[str, np.ndarray],
        evecs: dict[str, np.ndarray],
    ) -> float:
        return 0

    def grad(
        self,
        evals: dict[str, np.ndarray],
        evecs: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        to_return: dict[str, np.ndarray] = {}
        for axis in evals.keys():
            to_return[axis] = self.rhos[axis] * np.random.random(evals[axis].shape)
        return to_return