"""Typed, deterministic utilities used by the regression exercise notebook."""

from dataclasses import dataclass
import logging
from typing import Any, Sequence, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import regression_init


LOGGER = logging.getLogger(name=__name__)

FloatArray: TypeAlias = NDArray[np.float64]
IntegerArray: TypeAlias = NDArray[np.int8]
FeatureData: TypeAlias = pd.DataFrame | FloatArray
TargetData: TypeAlias = pd.Series | FloatArray

DEFAULT_COEFFICIENTS: FloatArray = np.asarray(
    a=[0.3, 0.5, -0.7],
    dtype=np.float64,
)


@dataclass(frozen=True)
class RegressionMetrics:
    """Regression metrics evaluated on one dataset."""

    mean_squared_error: float
    root_mean_squared_error: float
    r_squared: float


def _validated_coefficients(
    *,
    coefficients: ArrayLike | None,
) -> FloatArray:
    """Return a validated one-dimensional coefficient array."""
    coefficient_values = (
        DEFAULT_COEFFICIENTS.copy()
        if coefficients is None
        else np.asarray(a=coefficients, dtype=np.float64)
    )
    if coefficient_values.ndim != 1 or coefficient_values.shape[0] == 0:
        error_message = "coefficients must be a non-empty one-dimensional array"
        LOGGER.error(msg=error_message)
        raise ValueError(error_message)
    return coefficient_values


def _require_columns(
    *,
    data: pd.DataFrame,
    required_columns: Sequence[str],
) -> None:
    """Raise immediately when a required input column is absent."""
    missing_columns = [
        column_name
        for column_name in required_columns
        if column_name not in data.columns
    ]
    if missing_columns:
        error_message = f"Missing required columns: {missing_columns}"
        LOGGER.error(msg=error_message)
        raise KeyError(error_message)


def generate_linear_regression_data(
    *,
    number_of_samples: int = 1_000,
    random_seed: int = regression_init.RANDOM_SEED,
    coefficients: ArrayLike | None = None,
    intercept: float = 1.0,
) -> tuple[FloatArray, FloatArray]:
    """Generate the deterministic, noiseless linear-regression exercise data.

    Args:
        number_of_samples: Number of independent feature rows to generate.
        random_seed: NumPy random generator seed.
        coefficients: Optional one-dimensional coefficient vector.
        intercept: Constant added to every target value.

    Returns:
        A feature matrix and its exact linear target.

    Raises:
        ValueError: If the sample count or coefficient vector is invalid.
    """
    if number_of_samples <= 0:
        error_message = "number_of_samples must be positive"
        LOGGER.error(msg=error_message)
        raise ValueError(error_message)

    coefficient_values = _validated_coefficients(coefficients=coefficients)
    random_generator = np.random.default_rng(seed=random_seed)
    feature_data = random_generator.normal(
        loc=0.0,
        scale=1.0,
        size=(number_of_samples, coefficient_values.shape[0]),
    )
    target_data = feature_data @ coefficient_values + intercept
    return feature_data, target_data


def generate_logistic_regression_data(
    *,
    number_of_samples: int = 1_000,
    random_seed: int = regression_init.RANDOM_SEED,
    coefficients: ArrayLike | None = None,
    decision_threshold: float = 1.0,
) -> tuple[FloatArray, IntegerArray]:
    """Generate features and deterministic labels for the logistic exercise.

    Args:
        number_of_samples: Number of independent feature rows to generate.
        random_seed: NumPy random generator seed.
        coefficients: Optional one-dimensional boundary coefficient vector.
        decision_threshold: Score at or above which the label is one.

    Returns:
        A feature matrix and a binary target vector.
    """
    feature_data, linear_score = generate_linear_regression_data(
        number_of_samples=number_of_samples,
        random_seed=random_seed,
        coefficients=coefficients,
        intercept=0.0,
    )
    binary_target = (linear_score >= decision_threshold).astype(dtype=np.int8)
    return feature_data, binary_target


def build_polynomial_features(
    *,
    predictor_values: ArrayLike,
    maximum_degree: int = 49,
) -> FloatArray:
    """Create the columns x, x², ..., x**maximum_degree.

    Args:
        predictor_values: One-dimensional predictor values.
        maximum_degree: Highest polynomial power to include.

    Returns:
        A two-dimensional NumPy polynomial feature matrix.

    Raises:
        ValueError: If the degree is not positive or predictors are not 1-D.
    """
    if maximum_degree <= 0:
        error_message = "maximum_degree must be positive"
        LOGGER.error(msg=error_message)
        raise ValueError(error_message)

    predictor_array = np.asarray(a=predictor_values, dtype=np.float64)
    if predictor_array.ndim != 1:
        error_message = "predictor_values must be one-dimensional"
        LOGGER.error(msg=error_message)
        raise ValueError(error_message)

    predictor_column = predictor_array[:, np.newaxis]
    feature_transformer = PolynomialFeatures(
        degree=maximum_degree,
        include_bias=False,
        order="C",
    )
    return feature_transformer.fit_transform(X=predictor_column)


def select_numeric_house_data(
    *,
    house_data: pd.DataFrame,
) -> pd.DataFrame:
    """Select numeric house columns, remove the identifier, and fill NaNs.

    Args:
        house_data: Raw Ames house data.

    Returns:
        A copied numeric dataframe that includes ``SalePrice`` but not ``Id``.
    """
    _require_columns(
        data=house_data,
        required_columns=("Id", "SalePrice"),
    )
    numeric_house_data = house_data.select_dtypes(
        include="number",
    ).drop(
        columns=["Id"],
    )
    return numeric_house_data.fillna(value=0.0).copy(deep=True)


def add_one_hot_house_features(
    *,
    numeric_house_data: pd.DataFrame,
    house_data: pd.DataFrame,
    categorical_columns: Sequence[str],
) -> pd.DataFrame:
    """Append one-hot columns to an existing numeric house dataframe.

    Args:
        numeric_house_data: Numeric features aligned to ``house_data``.
        house_data: Raw Ames house data containing categorical columns.
        categorical_columns: Columns to encode.

    Returns:
        A new dataframe with first-level-dropped one-hot columns.
    """
    _require_columns(
        data=house_data,
        required_columns=categorical_columns,
    )
    categorical_column_names = [
        column_name for column_name in categorical_columns
    ]
    categorical_data = house_data.loc[:, categorical_column_names]
    encoded_features = pd.get_dummies(
        data=categorical_data,
        columns=categorical_column_names,
        drop_first=True,
        dtype=np.float64,
    )
    return pd.concat(
        objs=[numeric_house_data.copy(deep=True), encoded_features],
        axis=1,
    )


def add_required_house_features(
    *,
    house_data: pd.DataFrame,
) -> pd.DataFrame:
    """Create all explicitly requested house features and encodings.

    Args:
        house_data: Raw Ames house data.

    Returns:
        Numeric, encoded, and engineered features plus ``SalePrice``.
    """
    required_numeric_columns = (
        "LotArea",
        "1stFlrSF",
        "2ndFlrSF",
        "GarageArea",
        "BedroomAbvGr",
        "YearBuilt",
    )
    _require_columns(
        data=house_data,
        required_columns=required_numeric_columns,
    )
    numeric_house_data = select_numeric_house_data(house_data=house_data)
    prepared_house_data = add_one_hot_house_features(
        numeric_house_data=numeric_house_data,
        house_data=house_data,
        categorical_columns=("LotShape", "LandContour", "LotConfig"),
    )

    prepared_house_data.loc[:, "LotAreaSquareMeters"] = (
        prepared_house_data.loc[:, "LotArea"] * 0.092903
    )
    prepared_house_data.loc[:, "TotalFirstSecondFloorSF"] = (
        prepared_house_data.loc[:, "1stFlrSF"]
        + prepared_house_data.loc[:, "2ndFlrSF"]
    )
    prepared_house_data.loc[:, "GarageAreaSqrt"] = prepared_house_data.loc[
        :,
        "GarageArea",
    ].pow(other=0.5)
    prepared_house_data.loc[:, "LotAreaPerBedroom"] = (
        prepared_house_data.loc[:, "LotArea"]
        / (prepared_house_data.loc[:, "BedroomAbvGr"] + 1.0)
    )
    lot_area_by_year_mean = prepared_house_data.groupby(
        by="YearBuilt",
        sort=False,
    )["LotArea"].transform(func="mean")
    prepared_house_data.loc[:, "LotAreaToYearMean"] = (
        prepared_house_data.loc[:, "LotArea"]
        / (lot_area_by_year_mean + 1e-5)
    )
    prepared_house_data.loc[:, "LotAreaRank"] = prepared_house_data.loc[
        :,
        "LotArea",
    ].rank(
        method="first",
        ascending=False,
    )
    return prepared_house_data


def add_bonus_house_feature(
    *,
    house_data: pd.DataFrame,
) -> pd.DataFrame:
    """Add a quality-by-garage-capacity interaction to required features.

    Args:
        house_data: Raw Ames house data.

    Returns:
        Required features plus one interaction feature.
    """
    _require_columns(
        data=house_data,
        required_columns=("OverallQual", "GarageCars"),
    )
    prepared_house_data = add_required_house_features(house_data=house_data)
    prepared_house_data.loc[:, "OverallQualGarageCarsInteraction"] = (
        prepared_house_data.loc[:, "OverallQual"]
        * prepared_house_data.loc[:, "GarageCars"]
    )
    return prepared_house_data


def split_regression_data(
    *,
    feature_data: FeatureData,
    target_data: TargetData,
    training_fraction: float = 0.7,
    random_seed: int = regression_init.RANDOM_SEED,
) -> tuple[FeatureData, FeatureData, TargetData, TargetData]:
    """Create a deterministic shuffled regression train/test split.

    Args:
        feature_data: Feature rows.
        target_data: Regression targets aligned to the feature rows.
        training_fraction: Fraction assigned to the training segment.
        random_seed: Shuffle seed.

    Returns:
        Training features, testing features, training targets, testing targets.
    """
    if not 0.0 < training_fraction < 1.0:
        error_message = "training_fraction must be strictly between zero and one"
        LOGGER.error(msg=error_message)
        raise ValueError(error_message)

    # sklearn defines the data inputs as positional ``*arrays``; the typed
    # wrapper keeps every exercise call keyword-only.
    (
        training_features,
        testing_features,
        training_target,
        testing_target,
    ) = train_test_split(
        *(feature_data, target_data),
        train_size=training_fraction,
        random_state=random_seed,
        shuffle=True,
    )
    return (
        training_features,
        testing_features,
        training_target,
        testing_target,
    )


def fit_ridge_search(
    *,
    training_features: FeatureData,
    training_target: TargetData,
    alpha_values: Sequence[float],
    random_seed: int = regression_init.RANDOM_SEED,
    number_of_splits: int = 5,
    polynomial_degree: int | None = None,
) -> GridSearchCV:
    """Select Ridge alpha by shuffled cross-validated training MSE.

    Args:
        training_features: Model-training features.
        training_target: Model-training targets.
        alpha_values: Positive Ridge penalties to compare.
        random_seed: Cross-validation shuffle seed.
        number_of_splits: Number of cross-validation folds.
        polynomial_degree: Optional degree generated inside each fold.

    Returns:
        A fitted ``GridSearchCV`` containing the best preprocessing/model pipe.
    """
    alpha_array = np.asarray(
        a=[alpha_value for alpha_value in alpha_values],
        dtype=np.float64,
    )
    if (
        alpha_array.ndim != 1
        or alpha_array.shape[0] == 0
        or np.any(a=alpha_array <= 0.0)
    ):
        error_message = "alpha_values must contain positive values"
        LOGGER.error(msg=error_message)
        raise ValueError(error_message)
    if number_of_splits < 2:
        error_message = "number_of_splits must be at least two"
        LOGGER.error(msg=error_message)
        raise ValueError(error_message)

    pipeline_steps: list[tuple[str, Any]]
    if polynomial_degree is None:
        pipeline_steps = [
            ("scaler", StandardScaler()),
            ("ridge", Ridge()),
        ]
    else:
        pipeline_steps = [
            (
                "polynomial",
                PolynomialFeatures(
                    degree=polynomial_degree,
                    include_bias=False,
                    order="C",
                ),
            ),
            ("scaler", StandardScaler()),
            ("ridge", Ridge()),
        ]

    cross_validation = KFold(
        n_splits=number_of_splits,
        shuffle=True,
        random_state=random_seed,
    )
    model_search = GridSearchCV(
        estimator=Pipeline(steps=pipeline_steps),
        param_grid={"ridge__alpha": alpha_array},
        scoring="neg_mean_squared_error",
        cv=cross_validation,
        n_jobs=None,
        refit=True,
        return_train_score=True,
        error_score="raise",
    )
    model_search.fit(X=training_features, y=training_target)
    return model_search


def fit_knn_search(
    *,
    training_features: FeatureData,
    training_target: TargetData,
    neighbor_values: Sequence[int],
    random_seed: int = regression_init.RANDOM_SEED,
    number_of_splits: int = 5,
    weight_options: Sequence[str] = ("uniform", "distance"),
) -> GridSearchCV:
    """Select a scaled KNN regressor using training-only cross-validation.

    Args:
        training_features: Model-training features.
        training_target: Model-training targets.
        neighbor_values: Positive neighbor counts to compare.
        random_seed: Cross-validation shuffle seed.
        number_of_splits: Number of cross-validation folds.
        weight_options: KNN weighting strategies to compare.

    Returns:
        A fitted ``GridSearchCV`` containing the best scaled KNN model.
    """
    neighbor_array = np.asarray(
        a=[neighbor_value for neighbor_value in neighbor_values],
        dtype=np.int64,
    )
    if (
        neighbor_array.ndim != 1
        or neighbor_array.shape[0] == 0
        or np.any(a=neighbor_array <= 0)
    ):
        error_message = "neighbor_values must contain positive integers"
        LOGGER.error(msg=error_message)
        raise ValueError(error_message)
    if number_of_splits < 2:
        error_message = "number_of_splits must be at least two"
        LOGGER.error(msg=error_message)
        raise ValueError(error_message)

    cross_validation = KFold(
        n_splits=number_of_splits,
        shuffle=True,
        random_state=random_seed,
    )
    model_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor()),
        ],
    )
    model_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid={
            "knn__n_neighbors": neighbor_array,
            "knn__weights": [
                weight_option for weight_option in weight_options
            ],
            "knn__p": [1, 2],
        },
        scoring="neg_mean_squared_error",
        cv=cross_validation,
        n_jobs=None,
        refit=True,
        return_train_score=True,
        error_score="raise",
    )
    model_search.fit(X=training_features, y=training_target)
    return model_search


def evaluate_regressor(
    *,
    regression_model: RegressorMixin | Pipeline | GridSearchCV,
    feature_data: FeatureData,
    target_data: TargetData,
) -> RegressionMetrics:
    """Evaluate a fitted regressor using MSE, RMSE, and R².

    Args:
        regression_model: Fitted estimator exposing ``predict``.
        feature_data: Evaluation features.
        target_data: Observed evaluation targets.

    Returns:
        Immutable regression metrics.
    """
    prediction = regression_model.predict(X=feature_data)
    mean_squared_error_value = mean_squared_error(
        y_true=target_data,
        y_pred=prediction,
    )
    return RegressionMetrics(
        mean_squared_error=mean_squared_error_value,
        root_mean_squared_error=mean_squared_error_value**0.5,
        r_squared=r2_score(
            y_true=target_data,
            y_pred=prediction,
        ),
    )
