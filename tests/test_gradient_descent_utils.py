"""Numerical tests for the gradient-descent recitation utilities."""

from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
RECITATION_DIRECTORY = (
    PROJECT_DIRECTORY / "D - ML" / "regression" / "recitation"
)
sys.path[0:0] = [RECITATION_DIRECTORY.as_posix()]

import gradient_descent_init
from gradient_descent_utils import (
    build_linear_design_matrix,
    build_quadratic_design_matrix,
    calculate_linear_loss_surface,
    calculate_linear_mse,
    calculate_maximum_stable_learning_rate,
    fit_linear_regression_gradient_descent,
    fit_quadratic_regression_gradient_descent,
)


def _exercise_data() -> tuple[np.ndarray, np.ndarray]:
    """Return the deterministic noisy dataset used by the recitation."""
    feature_data, target_data = make_regression(
        n_samples=100,
        n_features=1,
        noise=15.0,
        random_state=gradient_descent_init.RANDOM_SEED,
    )
    return (
        np.asarray(a=feature_data, dtype=np.float64),
        np.asarray(a=target_data, dtype=np.float64),
    )


def test_linear_gradient_descent_matches_sklearn_solution() -> None:
    """Manual vectorized gradient descent must reach the OLS optimum."""
    feature_data, target_data = _exercise_data()
    gradient_result = fit_linear_regression_gradient_descent(
        feature_data=feature_data,
        target_data=target_data,
        initial_slope=10.0,
        initial_intercept=-10.0,
        number_of_epochs=250,
        learning_rate=0.3,
    )
    sklearn_model = LinearRegression()
    sklearn_model.fit(
        X=feature_data,
        y=target_data,
    )
    sklearn_prediction = sklearn_model.predict(
        X=feature_data,
    )
    expected_coefficients = np.asarray(
        a=[sklearn_model.coef_[0], sklearn_model.intercept_],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        actual=gradient_result.coefficients,
        desired=expected_coefficients,
        rtol=1e-9,
        atol=1e-9,
    )
    assert gradient_result.final_loss == pytest.approx(
        expected=mean_squared_error(
            y_true=target_data,
            y_pred=sklearn_prediction,
        ),
        rel=1e-12,
        abs=1e-12,
    )
    assert gradient_result.coefficient_history.shape == (251, 2)
    assert gradient_result.loss_history.shape == (251,)
    assert gradient_result.loss_history[-1] < gradient_result.loss_history[0]


def test_quadratic_gradient_descent_recovers_exercise_coefficients() -> None:
    """The bonus optimizer must recover a=3, b=-2, and c=15."""
    feature_data, _ = _exercise_data()
    predictor_values = feature_data[:, 0]
    polynomial_target = (
        3.0 * np.square(predictor_values)
        - 2.0 * predictor_values
        + 15.0
    )
    quadratic_design_matrix = build_quadratic_design_matrix(
        feature_data=feature_data,
    )
    stable_learning_rate = calculate_maximum_stable_learning_rate(
        design_matrix=quadratic_design_matrix,
    )
    gradient_result = fit_quadratic_regression_gradient_descent(
        feature_data=feature_data,
        target_data=polynomial_target,
        initial_quadratic_coefficient=10.0,
        initial_linear_coefficient=-10.0,
        initial_intercept=10.0,
        number_of_epochs=2_000,
        learning_rate=0.5 * stable_learning_rate,
    )

    np.testing.assert_allclose(
        actual=gradient_result.coefficients,
        desired=np.asarray(a=[3.0, -2.0, 15.0], dtype=np.float64),
        rtol=1e-8,
        atol=1e-8,
    )
    assert gradient_result.final_loss < 1e-16


def test_linear_loss_surface_matches_scalar_mse() -> None:
    """Every surface cell must use the same MSE definition as scalar cost."""
    feature_data, target_data = _exercise_data()
    slope_values = np.asarray(a=[-1.0, 0.0, 1.0], dtype=np.float64)
    intercept_values = np.asarray(a=[-2.0, 0.0, 2.0], dtype=np.float64)
    loss_surface = calculate_linear_loss_surface(
        feature_data=feature_data,
        target_data=target_data,
        slope_values=slope_values,
        intercept_values=intercept_values,
    )
    scalar_loss = calculate_linear_mse(
        feature_data=feature_data,
        target_data=target_data,
        slope=1.0,
        intercept=0.0,
    )

    assert loss_surface.shape == (3, 3)
    assert loss_surface[1, 2] == pytest.approx(
        expected=scalar_loss,
        rel=1e-12,
        abs=1e-12,
    )


def test_maximum_stable_learning_rate_matches_hessian_eigenvalue() -> None:
    """The reported fixed-step limit must equal 2/lambda_max(H)."""
    feature_data, _ = _exercise_data()
    design_matrix = build_linear_design_matrix(
        feature_data=feature_data,
    )
    hessian_matrix = (
        2.0
        / design_matrix.shape[0]
        * design_matrix.transpose()
        @ design_matrix
    )
    maximum_eigenvalue = np.linalg.eigvalsh(
        a=hessian_matrix,
    )[-1]

    assert calculate_maximum_stable_learning_rate(
        design_matrix=design_matrix,
    ) == pytest.approx(
        expected=2.0 / maximum_eigenvalue,
        rel=1e-12,
        abs=1e-12,
    )


def test_invalid_shapes_fail_immediately() -> None:
    """Mismatched rows and invalid feature shapes must raise clear errors."""
    with pytest.raises(expected_exception=ValueError):
        fit_linear_regression_gradient_descent(
            feature_data=np.asarray(a=[[1.0], [2.0]], dtype=np.float64),
            target_data=np.asarray(a=[1.0], dtype=np.float64),
        )

    with pytest.raises(expected_exception=ValueError):
        calculate_linear_mse(
            feature_data=np.asarray(
                a=[[1.0, 2.0], [3.0, 4.0]],
                dtype=np.float64,
            ),
            target_data=np.asarray(a=[1.0, 2.0], dtype=np.float64),
            slope=1.0,
            intercept=0.0,
        )
