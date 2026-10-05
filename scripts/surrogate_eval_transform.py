"""Target transforms for the surrogate evaluation harness.

The AmpC target ``y`` is a probability of binding with mean 0.0396 and standard
deviation 0.0218 but a maximum of 0.666, so the high-scoring molecules sit nearly 30
standard deviations out in the standardized space the GP fits. A Gaussian likelihood
with a single noise level, fitted against 10M bulk rows, underpredicts them. These
transforms fit the GP on ``log(y)`` or ``logit(y)`` instead, which pulls that tail in.

The GP therefore predicts a Gaussian over transformed targets. Metrics stay on the
``y`` scale so they compare with the untransformed runs, which means mapping that
Gaussian back through a nonlinear function: the mean of the back-transformed
distribution is *not* the back-transformed mean. :meth:`TargetTransform.
posterior_moments` integrates the mapped Gaussian with Gauss-Hermite quadrature, which
is exact for the identity and accurate to many digits for the other two, and uses the
same code path for both transforms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

#: Quadrature nodes used to map a Gaussian posterior through the inverse transform.
#: Gauss-Hermite converges geometrically for the smooth integrands here, so 32 nodes
#: are far more than the metrics need.
QUADRATURE_NODES = 32


class TargetTransform(ABC):
    """A monotone map applied to the training targets before the GP fit."""

    #: Name used in the CLI, the run config and the fit summary.
    name: str

    @abstractmethod
    def forward(self, y: np.ndarray) -> np.ndarray:
        """Map targets to the space the GP fits.

        Parameters
        ----------
        y : np.ndarray
            Targets on the original scale.

        Returns
        -------
        np.ndarray
            Transformed targets.
        """

    @abstractmethod
    def inverse(self, z: np.ndarray) -> np.ndarray:
        """Map transformed values back to the original scale.

        Parameters
        ----------
        z : np.ndarray
            Values in the transformed space.

        Returns
        -------
        np.ndarray
            Values on the original target scale.
        """

    def validate(self, y: np.ndarray) -> None:
        """Raise if any target lies outside the transform's domain.

        Parameters
        ----------
        y : np.ndarray
            Targets on the original scale.
        """

    def posterior_moments(
        self, mean: np.ndarray, std: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map a Gaussian posterior in transformed space to the original scale.

        Parameters
        ----------
        mean : np.ndarray
            Posterior means in the transformed space.
        std : np.ndarray
            Posterior standard deviations in the transformed space.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Mean and standard deviation of the back-transformed distribution.
        """
        mu = np.asarray(mean, dtype=np.float64).reshape(-1)
        sigma = np.asarray(std, dtype=np.float64).reshape(-1)
        nodes, weights = np.polynomial.hermite.hermgauss(QUADRATURE_NODES)
        # z = mu + sqrt(2) * sigma * node turns the Gauss-Hermite weight exp(-t^2)
        # into the Gaussian density, up to the 1/sqrt(pi) below.
        grid = self.inverse(
            mu[:, None] + np.sqrt(2.0) * sigma[:, None] * nodes[None, :]
        )
        probabilities = weights / np.sqrt(np.pi)
        first = grid @ probabilities
        # E[g^2] - E[g]^2 cancels catastrophically once the posterior is narrow, which
        # is the usual case here, and leaves a spurious floor under the standard
        # deviation. Centering first costs one more pass and cannot go negative.
        variance = ((grid - first[:, None]) ** 2) @ probabilities
        return first, np.sqrt(variance)


class IdentityTransform(TargetTransform):
    """The untransformed target, leaving the fit exactly as it was."""

    name = "none"

    def forward(self, y: np.ndarray) -> np.ndarray:
        """Return the targets unchanged."""
        return np.asarray(y, dtype=np.float64)

    def inverse(self, z: np.ndarray) -> np.ndarray:
        """Return the values unchanged."""
        return np.asarray(z, dtype=np.float64)

    def posterior_moments(
        self, mean: np.ndarray, std: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the posterior unchanged, bypassing the quadrature.

        Parameters
        ----------
        mean : np.ndarray
            Posterior means.
        std : np.ndarray
            Posterior standard deviations.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The inputs as float arrays, so an untransformed run reproduces the
            earlier numbers exactly rather than to quadrature accuracy.
        """
        return (
            np.asarray(mean, dtype=np.float64).reshape(-1),
            np.asarray(std, dtype=np.float64).reshape(-1),
        )


class LogTransform(TargetTransform):
    """``log(y)``, for a positive target with a long right tail."""

    name = "log"

    def forward(self, y: np.ndarray) -> np.ndarray:
        """Return ``log(y)``."""
        return np.log(np.asarray(y, dtype=np.float64))

    def inverse(self, z: np.ndarray) -> np.ndarray:
        """Return ``exp(z)``."""
        return np.exp(np.asarray(z, dtype=np.float64))

    def validate(self, y: np.ndarray) -> None:
        """Raise if any target is not strictly positive.

        Parameters
        ----------
        y : np.ndarray
            Targets on the original scale.

        Raises
        ------
        ValueError
            If any target is zero or negative.
        """
        values = np.asarray(y, dtype=np.float64)
        bad = int(np.sum(~(values > 0.0)))
        if bad:
            raise ValueError(
                f"log needs strictly positive targets, but {bad} of {values.size} "
                f"are not (minimum {values.min()})."
            )


class LogitTransform(TargetTransform):
    """``logit(y)``, for a target that is a probability."""

    name = "logit"

    def forward(self, y: np.ndarray) -> np.ndarray:
        """Return ``log(y / (1 - y))``."""
        values = np.asarray(y, dtype=np.float64)
        return np.log(values) - np.log1p(-values)

    def inverse(self, z: np.ndarray) -> np.ndarray:
        """Return the logistic function of ``z``, evaluated without overflow."""
        values = np.asarray(z, dtype=np.float64)
        # exp(-z) overflows for very negative z and exp(z) for very positive z, so
        # each half uses the form that only ever exponentiates a negative number.
        positive = values >= 0.0
        out = np.empty_like(values)
        out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
        exponentiated = np.exp(values[~positive])
        out[~positive] = exponentiated / (1.0 + exponentiated)
        return out

    def validate(self, y: np.ndarray) -> None:
        """Raise if any target lies outside the open interval ``(0, 1)``.

        Parameters
        ----------
        y : np.ndarray
            Targets on the original scale.

        Raises
        ------
        ValueError
            If any target is not strictly between zero and one.
        """
        values = np.asarray(y, dtype=np.float64)
        bad = int(np.sum(~((values > 0.0) & (values < 1.0))))
        if bad:
            raise ValueError(
                f"logit needs targets strictly inside (0, 1), but {bad} of "
                f"{values.size} are not (range {values.min()} to {values.max()})."
            )


#: The transforms the harness accepts, keyed by their CLI name.
TARGET_TRANSFORMS: dict[str, TargetTransform] = {
    transform.name: transform
    for transform in (IdentityTransform(), LogTransform(), LogitTransform())
}


def build_target_transform(name: str) -> TargetTransform:
    """Return the transform registered under ``name``.

    Parameters
    ----------
    name : str
        One of the keys of :data:`TARGET_TRANSFORMS`.

    Returns
    -------
    TargetTransform
        The transform.

    Raises
    ------
    ValueError
        If no transform is registered under that name.
    """
    if name not in TARGET_TRANSFORMS:
        raise ValueError(
            f"Unknown target transform {name!r}; have {sorted(TARGET_TRANSFORMS)}."
        )
    return TARGET_TRANSFORMS[name]
