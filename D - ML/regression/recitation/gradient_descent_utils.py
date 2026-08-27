"""Typed NumPy utilities for the gradient-descent recitation."""

from dataclasses import dataclass
import logging
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

import gradient_descent_init


LOGGER = logging.getLogger(name=__name__)

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class GradientDescentResult:
    """Parameters and complete optimization history for least squares."""

    coefficients: FloatArray
    coefficient_history: FloatArray
    loss_history: FloatArray
    learning_rate: float
    number_of_epochs: int

    @property
    def final_loss(self) -> float:
        """Return the MSE evaluated after the final parameter update."""
        return float(self.loss_history[-1])


def _as_single_feature_vector(
    *,
    feature_data: ArrayLike,
) -> FloatArray:
    """Return one finite predictor column as a one-dimensional array."""
    feature_array = np.asarray(
        a=feature_data,
        dtype=np.float64,
    )
    if feature_array.ndim == 2 and feature_array.shape[1] == 1:
        feature_array = feature_array[:, 0]
    if feature_array.ndim != 1 or feature_array.shape[0] == 0:
        error_message = (
            "feature_data must contain one non-empty predictor column"
        )
        LOGGER.error(msg=error_message)
        raise ValueError(error_message)
    if not np.all(a=np.isfinite(feature_array)):
        error_message = "feature_data must contain only finite values"
        LOGGER.error(msg=error_message)
        raise ValueError(error_message)
    return feature_array


def _as_target_vector(
    *,
    target_data: ArrayLike,
) -> FloatArray:
    """Return a finite, non-empty one-dimensional target array."""
    target_array = np.asarray(
        a=target_data,
        dtype=np.float64,
    )
    if target_array.ndim == 2 and target_array.shape[1] == 1:
        target_array = target_array[:, 0]
    if target_array.ndim != 1 or target_array.shape[0] == 0:
        error_message = "target_data must be a non-empty one-dimensional array"
        LOGGER.error(msg=error_message)
        raise ValueError(error_message)
    if not np.all(a=np.isfinite(target_array)):
        error_message = "target_data must contain only finite values"
        LOGGER.error(msg=error_message)
        raise ValueError(error_message)
    return target_array


def _as_design_matrix(
    *,
    design_matrix: ArrayLike,
) -> FloatArray:
    """Return a validated finite two-dimensional design matrix."""
    design_array = np.asarray(
        a=design_matrix,
        dtype=np.float64,
    )
    if (
        design_array.ndim != 2
        or design_array.shape[0] == 0
        or design_array.shape[1] == 0
    ):
        error_message = "design_matrix must be a non-empty 2-D array"
        LOGGER.error(msg=error_message)
        raise ValueError(error_message)
    if not np.all(a=np.isfinite(design_array)):
        error_message = "design_matrix must contain only finite values"
        LOGGER.error(msg=error_message)
        raise ValueError(error_message)
    return design_array


def _validated_training_data(
    *,
    design_matrix: ArrayLike,
    target_data: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """Validate and align a design matrix with its target vector."""
    design_array = _as_design_matrix(
        design_matrix=design_matrix,
    )
    target_array = _as_target_vector(
        target_data=target_data,
    )
    if design_array.shape[0] != target_array.shape[0]:
        error_message = (
            "design_matrix and target_data must contain the same row count"
        )
        LOGGER.error(msg=error_message)
        raise ValueError(error_message)
    return design_array, target_array


def _initial_coefficient_vector(
    *,
    initial_coefficients: ArrayLike | None,
    number_of_parameters: int,
) -> FloatArray:
    """Return validated starting coefficients for an optimization."""
    coefficient_array = (
        np.zeros(
            shape=number_of_parameters,
            dtype=np.float64,
        )
        if initial_coefficients is None
        else np.asarray(
            a=initial_coefficients,
            dtype=np.float64,
        )
    )
    if coefficient_array.shape != (number_of_parameters,):
        error_message = (
            "initial_coefficients must have one value per design column"
        )
        LOGGER.error(msg=error_message)
        raise ValueError(error_message)
    if not np.all(a=np.isfinite(coefficient_array)):
        error_message = "initial_coefficients must contain only finite values"
        LOGGER.error(msg=error_message)
        raise ValueError(error_message)
    return coefficient_array.copy()


def build_linear_design_matrix(
    *,
    feature_data: ArrayLike,
) -> FloatArray:
    """Build columns ``[x, 1]`` for slope and intercept estimation.

    Args:
        feature_data: One predictor column.

    Returns:
        A two-column design matrix ordered as slope then intercept.
    """
    predictor_values = _as_single_feature_vector(
        feature_data=feature_data,
    )
    return np.column_stack(
        tup=(
            predictor_values,
            np.ones(
                shape=predictor_values.shape[0],
                dtype=np.float64,
            ),
        ),
    )


def build_quadratic_design_matrix(
    *,
    feature_data: ArrayLike,
) -> FloatArray:
    """Build columns ``[x**2, x, 1]`` for ``a``, ``b``, and ``c``.

    Args:
        feature_data: One predictor column.

    Returns:
        A three-column quadratic design matrix.
    """
    predictor_values = _as_single_feature_vector(
        feature_data=feature_data,
    )
    return np.column_stack(
        tup=(
            np.square(predictor_values),
            predictor_values,
            np.ones(
                shape=predictor_values.shape[0],
                dtype=np.float64,
            ),
        ),
    )


def calculate_least_squares_mse(
    *,
    design_matrix: ArrayLike,
    target_data: ArrayLike,
    coefficients: ArrayLike,
) -> float:
    """Calculate mean squared error for an arbitrary linear design.

    Args:
        design_matrix: Numeric matrix with one column per coefficient.
        target_data: Observed target values.
        coefficients: Coefficients aligned to the design columns.

    Returns:
        Mean squared residual error.
    """
    design_array, target_array = _validated_training_data(
        design_matrix=design_matrix,
        target_data=target_data,
    )
    coefficient_array = _initial_coefficient_vector(
        initial_coefficients=coefficients,
        number_of_parameters=design_array.shape[1],
    )
    residual_values = design_array @ coefficient_array - target_array
    return float(
        np.mean(
            a=np.square(residual_values),
        )
    )


def fit_least_squares_gradient_descent(
    *,
    design_matrix: ArrayLike,
    target_data: ArrayLike,
    initial_coefficients: ArrayLike | None = None,
    number_of_epochs: int = 10_000,
    learning_rate: float = 0.0001,
) -> GradientDescentResult:
    """Minimize MSE with full-batch vectorized gradient descent.

    Args:
        design_matrix: Numeric matrix with one column per coefficient.
        target_data: Observed target vector.
        initial_coefficients: Optional starting parameter vector.
        number_of_epochs: Number of complete gradient updates.
        learning_rate: Fixed positive gradient step size.

    Returns:
        Final coefficients and aligned parameter/loss histories.

    Raises:
        ValueError: If inputs or optimization settings are invalid.
        FloatingPointError: If an update produces a non-finite value.
    """
    design_array, target_array = _validated_training_data(
        design_matrix=design_matrix,
        target_data=target_data,
    )
    if number_of_epochs <= 0:
        error_message = "number_of_epochs must be positive"
        LOGGER.error(msg=error_message)
        raise ValueError(error_message)
    if not np.isfinite(learning_rate) or learning_rate <= 0.0:
        error_message = "learning_rate must be positive and finite"
        LOGGER.error(msg=error_message)
        raise ValueError(error_message)

    current_coefficients = _initial_coefficient_vector(
        initial_coefficients=initial_coefficients,
        number_of_parameters=design_array.shape[1],
    )
    coefficient_history = np.empty(
        shape=(number_of_epochs + 1, design_array.shape[1]),
        dtype=np.float64,
    )
    loss_history = np.empty(
        shape=number_of_epochs + 1,
        dtype=np.float64,
    )
    coefficient_history[0, :] = current_coefficients
    initial_residuals = design_array @ current_coefficients - target_array
    loss_history[0] = np.mean(
        a=np.square(initial_residuals),
    )

    number_of_observations = float(design_array.shape[0])
    for epoch_index in range(number_of_epochs):
        current_residuals = design_array @ current_coefficients - target_array
        current_gradient = (
            2.0
            / number_of_observations
            * (design_array.transpose() @ current_residuals)
        )
        current_coefficients = (
            current_coefficients - learning_rate * current_gradient
        )
        if not np.all(a=np.isfinite(current_coefficients)):
            error_message = (
                f"Gradient descent produced non-finite coefficients at "
                f"epoch {epoch_index + 1}"
            )
            LOGGER.error(msg=error_message)
            raise FloatingPointError(error_message)

        updated_residuals = design_array @ current_coefficients - target_array
        updated_loss = np.mean(
            a=np.square(updated_residuals),
        )
        if not np.isfinite(updated_loss):
            error_message = (
                f"Gradient descent produced non-finite loss at "
                f"epoch {epoch_index + 1}"
            )
            LOGGER.error(msg=error_message)
            raise FloatingPointError(error_message)

        coefficient_history[epoch_index + 1, :] = current_coefficients
        loss_history[epoch_index + 1] = updated_loss

    return GradientDescentResult(
        coefficients=current_coefficients.copy(),
        coefficient_history=coefficient_history,
        loss_history=loss_history,
        learning_rate=float(learning_rate),
        number_of_epochs=number_of_epochs,
    )


def fit_linear_regression_gradient_descent(
    *,
    feature_data: ArrayLike,
    target_data: ArrayLike,
    initial_slope: float = 0.0,
    initial_intercept: float = 0.0,
    number_of_epochs: int = 10_000,
    learning_rate: float = 0.0001,
) -> GradientDescentResult:
    """Fit ``y = m*x + b`` using vectorized gradient descent.

    Args:
        feature_data: One predictor column.
        target_data: Observed target values.
        initial_slope: Starting value for ``m``.
        initial_intercept: Starting value for ``b``.
        number_of_epochs: Number of full-batch updates.
        learning_rate: Fixed positive step size.

    Returns:
        Gradient descent result ordered as ``[m, b]``.
    """
    design_matrix = build_linear_design_matrix(
        feature_data=feature_data,
    )
    return fit_least_squares_gradient_descent(
        design_matrix=design_matrix,
        target_data=target_data,
        initial_coefficients=np.asarray(
            a=[initial_slope, initial_intercept],
            dtype=np.float64,
        ),
        number_of_epochs=number_of_epochs,
        learning_rate=learning_rate,
    )


def fit_quadratic_regression_gradient_descent(
    *,
    feature_data: ArrayLike,
    target_data: ArrayLike,
    initial_quadratic_coefficient: float = 0.0,
    initial_linear_coefficient: float = 0.0,
    initial_intercept: float = 0.0,
    number_of_epochs: int = 10_000,
    learning_rate: float = 0.0001,
) -> GradientDescentResult:
    """Fit ``y = a*x**2 + b*x + c`` with vectorized gradient descent.

    Args:
        feature_data: One predictor column.
        target_data: Observed polynomial target values.
        initial_quadratic_coefficient: Starting value for ``a``.
        initial_linear_coefficient: Starting value for ``b``.
        initial_intercept: Starting value for ``c``.
        number_of_epochs: Number of full-batch updates.
        learning_rate: Fixed positive step size.

    Returns:
        Gradient descent result ordered as ``[a, b, c]``.
    """
    design_matrix = build_quadratic_design_matrix(
        feature_data=feature_data,
    )
    return fit_least_squares_gradient_descent(
        design_matrix=design_matrix,
        target_data=target_data,
        initial_coefficients=np.asarray(
            a=[
                initial_quadratic_coefficient,
                initial_linear_coefficient,
                initial_intercept,
            ],
            dtype=np.float64,
        ),
        number_of_epochs=number_of_epochs,
        learning_rate=learning_rate,
    )


def calculate_linear_mse(
    *,
    feature_data: ArrayLike,
    target_data: ArrayLike,
    slope: float,
    intercept: float,
) -> float:
    """Calculate MSE for one ``y = m*x + b`` parameter pair.

    Args:
        feature_data: One predictor column.
        target_data: Observed target values.
        slope: Candidate ``m`` value.
        intercept: Candidate ``b`` value.

    Returns:
        Mean squared error for the candidate line.
    """
    return calculate_least_squares_mse(
        design_matrix=build_linear_design_matrix(
            feature_data=feature_data,
        ),
        target_data=target_data,
        coefficients=np.asarray(
            a=[slope, intercept],
            dtype=np.float64,
        ),
    )


def calculate_linear_loss_surface(
    *,
    feature_data: ArrayLike,
    target_data: ArrayLike,
    slope_values: ArrayLike,
    intercept_values: ArrayLike,
) -> FloatArray:
    """Calculate a full MSE grid without Python parameter loops.

    Args:
        feature_data: One predictor column.
        target_data: Observed target values.
        slope_values: Horizontal grid values for ``m``.
        intercept_values: Vertical grid values for ``b``.

    Returns:
        A matrix shaped ``(number of intercepts, number of slopes)``.
    """
    predictor_array = _as_single_feature_vector(
        feature_data=feature_data,
    )
    target_array = _as_target_vector(
        target_data=target_data,
    )
    if predictor_array.shape[0] != target_array.shape[0]:
        error_message = (
            "feature_data and target_data must contain the same row count"
        )
        LOGGER.error(msg=error_message)
        raise ValueError(error_message)

    slope_array = np.asarray(
        a=slope_values,
        dtype=np.float64,
    )
    intercept_array = np.asarray(
        a=intercept_values,
        dtype=np.float64,
    )
    if (
        slope_array.ndim != 1
        or slope_array.shape[0] == 0
        or intercept_array.ndim != 1
        or intercept_array.shape[0] == 0
    ):
        error_message = (
            "slope_values and intercept_values must be non-empty 1-D arrays"
        )
        LOGGER.error(msg=error_message)
        raise ValueError(error_message)
    if (
        not np.all(a=np.isfinite(slope_array))
        or not np.all(a=np.isfinite(intercept_array))
    ):
        error_message = "loss-surface grid values must be finite"
        LOGGER.error(msg=error_message)
        raise ValueError(error_message)

    slope_grid = slope_array[np.newaxis, :]
    intercept_grid = intercept_array[:, np.newaxis]
    mean_predictor = np.mean(a=predictor_array)
    mean_target = np.mean(a=target_array)
    mean_predictor_squared = np.mean(
        a=np.square(predictor_array),
    )
    mean_target_squared = np.mean(
        a=np.square(target_array),
    )
    mean_predictor_target = np.mean(
        a=predictor_array * target_array,
    )
    return (
        np.square(slope_grid) * mean_predictor_squared
        + np.square(intercept_grid)
        + mean_target_squared
        + 2.0 * slope_grid * intercept_grid * mean_predictor
        - 2.0 * slope_grid * mean_predictor_target
        - 2.0 * intercept_grid * mean_target
    )


def calculate_maximum_stable_learning_rate(
    *,
    design_matrix: ArrayLike,
) -> float:
    """Return the strict upper fixed-step bound ``2/lambda_max(H)``.

    Args:
        design_matrix: Linear least-squares design matrix.

    Returns:
        Theoretical upper learning-rate bound for this quadratic MSE.
    """
    design_array = _as_design_matrix(
        design_matrix=design_matrix,
    )
    hessian_matrix = (
        2.0
        / float(design_array.shape[0])
        * (design_array.transpose() @ design_array)
    )
    maximum_eigenvalue = np.linalg.eigvalsh(
        a=hessian_matrix,
    )[-1]
    if maximum_eigenvalue <= 0.0:
        error_message = "design_matrix does not define a positive curvature"
        LOGGER.error(msg=error_message)
        raise ValueError(error_message)
    return float(2.0 / maximum_eigenvalue)
