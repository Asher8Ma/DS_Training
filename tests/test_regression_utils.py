"""Focused tests for the regression exercise utilities."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
EXERCISE_DIRECTORY = PROJECT_DIRECTORY / "D - ML" / "regression" / "exercise"
sys.path[0:0] = [EXERCISE_DIRECTORY.as_posix()]

from regression_init import RANDOM_SEED, RESOURCE_DIRECTORY
from regression_utils import (
    DEFAULT_COEFFICIENTS,
    add_bonus_house_feature,
    add_one_hot_house_features,
    add_required_house_features,
    build_polynomial_features,
    evaluate_regressor,
    fit_knn_search,
    fit_ridge_search,
    generate_linear_regression_data,
    generate_logistic_regression_data,
    select_numeric_house_data,
    split_regression_data,
)


def test_linear_data_recovers_exact_coefficients(
    *,
    number_of_samples: int = 1_000,
) -> None:
    """The deterministic noiseless generator must recover its known equation."""
    feature_data, target_data = generate_linear_regression_data(
        number_of_samples=number_of_samples,
        random_seed=RANDOM_SEED,
    )
    regression_model = LinearRegression()
    regression_model.fit(X=feature_data, y=target_data)

    assert feature_data.shape == (number_of_samples, DEFAULT_COEFFICIENTS.shape[0])
    np.testing.assert_allclose(
        actual=regression_model.coef_,
        desired=DEFAULT_COEFFICIENTS,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        actual=regression_model.intercept_,
        desired=1.0,
        rtol=0.0,
        atol=1e-12,
    )


def test_logistic_data_matches_the_requested_boundary(
    *,
    number_of_samples: int = 1_000,
    decision_threshold: float = 1.0,
) -> None:
    """Generated labels must equal the exercise's deterministic inequality."""
    feature_data, binary_target = generate_logistic_regression_data(
        number_of_samples=number_of_samples,
        random_seed=RANDOM_SEED,
        decision_threshold=decision_threshold,
    )
    expected_target = (
        feature_data @ DEFAULT_COEFFICIENTS >= decision_threshold
    ).astype(dtype=np.int8)

    np.testing.assert_array_equal(
        actual=binary_target,
        desired=expected_target,
    )
    assert np.unique(ar=binary_target).shape[0] == 2


def test_seeded_logistic_datasets_are_nested(
    *,
    smaller_sample_count: int = 1_000,
    larger_sample_count: int = 10_000,
    decision_threshold: float = 1.0,
) -> None:
    """The larger seeded experiment must begin with the smaller experiment."""
    smaller_features, smaller_target = generate_logistic_regression_data(
        number_of_samples=smaller_sample_count,
        random_seed=RANDOM_SEED,
        decision_threshold=decision_threshold,
    )
    larger_features, larger_target = generate_logistic_regression_data(
        number_of_samples=larger_sample_count,
        random_seed=RANDOM_SEED,
        decision_threshold=decision_threshold,
    )

    np.testing.assert_array_equal(
        actual=larger_features[:smaller_sample_count],
        desired=smaller_features,
    )
    np.testing.assert_array_equal(
        actual=larger_target[:smaller_sample_count],
        desired=smaller_target,
    )


def test_polynomial_features_contain_powers_one_through_forty_nine(
    *,
    maximum_degree: int = 49,
) -> None:
    """The ready-made transformer must reproduce the requested power basis."""
    function_training_data = pd.read_csv(
        filepath_or_buffer=RESOURCE_DIRECTORY / "func_1_train.csv",
    )
    predictor_values = function_training_data.loc[:, "x"].to_numpy(
        dtype=np.float64,
    )
    polynomial_features = build_polynomial_features(
        predictor_values=predictor_values,
        maximum_degree=maximum_degree,
    )

    assert polynomial_features.shape == (
        function_training_data.shape[0],
        maximum_degree,
    )
    np.testing.assert_allclose(
        actual=polynomial_features[:, 0],
        desired=predictor_values,
    )
    np.testing.assert_allclose(
        actual=polynomial_features[:, -1],
        desired=predictor_values**maximum_degree,
    )


def test_house_preparation_uses_real_data_and_creates_required_features(
    *,
    square_feet_to_square_meters: float = 0.092903,
) -> None:
    """Numeric selection, one-hot encoding, and formulas must be correct."""
    house_data = pd.read_csv(
        filepath_or_buffer=RESOURCE_DIRECTORY / "house_data.csv",
    )
    numeric_house_data = select_numeric_house_data(house_data=house_data)
    encoded_house_data = add_one_hot_house_features(
        numeric_house_data=numeric_house_data,
        house_data=house_data,
        categorical_columns=("LotShape", "LandContour"),
    )
    engineered_house_data = add_required_house_features(house_data=house_data)
    bonus_house_data = add_bonus_house_feature(house_data=house_data)

    assert "Id" not in numeric_house_data.columns
    assert "SalePrice" in numeric_house_data.columns
    assert numeric_house_data.isna().sum().sum() == 0
    assert encoded_house_data.shape[1] > numeric_house_data.shape[1]
    assert any(
        column_name.startswith("LotShape_")
        for column_name in encoded_house_data.columns
    )
    assert any(
        column_name.startswith("LandContour_")
        for column_name in encoded_house_data.columns
    )
    assert any(
        column_name.startswith("LotConfig_")
        for column_name in engineered_house_data.columns
    )

    first_row_index = engineered_house_data.index[0]
    np.testing.assert_allclose(
        actual=engineered_house_data.loc[
            first_row_index,
            "LotAreaSquareMeters",
        ],
        desired=house_data.loc[first_row_index, "LotArea"]
        * square_feet_to_square_meters,
    )
    np.testing.assert_allclose(
        actual=engineered_house_data.loc[
            first_row_index,
            "TotalFirstSecondFloorSF",
        ],
        desired=house_data.loc[first_row_index, "1stFlrSF"]
        + house_data.loc[first_row_index, "2ndFlrSF"],
    )
    np.testing.assert_allclose(
        actual=engineered_house_data.loc[first_row_index, "GarageAreaSqrt"],
        desired=house_data.loc[first_row_index, "GarageArea"] ** 0.5,
    )
    largest_lot_index = engineered_house_data.loc[:, "LotArea"].idxmax()
    assert engineered_house_data.loc[largest_lot_index, "LotAreaRank"] == 1.0
    np.testing.assert_allclose(
        actual=bonus_house_data.loc[
            first_row_index,
            "OverallQualGarageCarsInteraction",
        ],
        desired=house_data.loc[first_row_index, "OverallQual"]
        * house_data.loc[first_row_index, "GarageCars"],
    )


def test_splits_and_model_searches_are_deterministic_on_real_data(
    *,
    training_fraction: float = 0.7,
) -> None:
    """The shared split, Ridge search, KNN search, and metrics must work."""
    function_training_data = pd.read_csv(
        filepath_or_buffer=RESOURCE_DIRECTORY / "func_1_train.csv",
    )
    predictor_values = function_training_data.loc[:, "x"].to_numpy(
        dtype=np.float64,
    )
    polynomial_features = build_polynomial_features(
        predictor_values=predictor_values,
        maximum_degree=5,
    )
    target_values = function_training_data.loc[:, "y"].to_numpy(
        dtype=np.float64,
    )
    (
        training_features,
        testing_features,
        training_target,
        testing_target,
    ) = split_regression_data(
        feature_data=polynomial_features,
        target_data=target_values,
        training_fraction=training_fraction,
        random_seed=RANDOM_SEED,
    )
    ridge_search = fit_ridge_search(
        training_features=training_features,
        training_target=training_target,
        alpha_values=(0.01, 0.1, 1.0),
        random_seed=RANDOM_SEED,
        number_of_splits=3,
    )
    knn_search = fit_knn_search(
        training_features=training_features,
        training_target=training_target,
        neighbor_values=(3, 5),
        random_seed=RANDOM_SEED,
        number_of_splits=3,
        weight_options=("uniform", "distance"),
    )
    ridge_metrics = evaluate_regressor(
        regression_model=ridge_search,
        feature_data=testing_features,
        target_data=testing_target,
    )
    knn_metrics = evaluate_regressor(
        regression_model=knn_search,
        feature_data=testing_features,
        target_data=testing_target,
    )

    assert training_features.shape[0] == 140
    assert testing_features.shape[0] == 60
    assert ridge_search.best_params_["ridge__alpha"] in (0.01, 0.1, 1.0)
    assert knn_search.best_params_["knn__n_neighbors"] in (3, 5)
    assert ridge_metrics.mean_squared_error >= 0.0
    assert ridge_metrics.root_mean_squared_error >= 0.0
    assert np.isfinite(ridge_metrics.r_squared)
    assert knn_metrics.mean_squared_error >= 0.0
