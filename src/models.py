from typing import Tuple
import pandas as pd

from darts import TimeSeries
from darts.models import (
    RegressionModel,
    NaiveSeasonal,
    )
from darts.utils.statistics import check_seasonality
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LinearRegression
from darts.metrics import mape
from dataclasses import dataclass


@dataclass
class ModelLags:
    """
    Lags for Darts lagged model class
    """
    num_lags: list[int]
    num_lags_past_covariates: list[int]
    num_lags_future_covariates: list[int]


MODEL_LAGS = ModelLags(
    num_lags=[-3, -2],
    num_lags_past_covariates=[-4, -3],
    num_lags_future_covariates=[0]
)


def fit_regression_model(
    targets: list[TimeSeries],
    past_features: list[TimeSeries],
    future_features: list[TimeSeries],
    model_class: Pipeline | None = None,
    model_lags: ModelLags | None = None,
) -> RegressionModel:
    """
    Initialise and fit a Darts model

    Args:
        targets (dict[int, TimeSeries]): Target timeseries
        past_features (list[TimeSeries]): Past features timeseries
        future_features (list[TimeSeries]): Future features timeseries
        model_class (Pipeline | None, optional): Model class as a scikitlearn pipeline. Defaults to None.
        model_lags (list[float], optional): Model lags for target, past and future features. Defaults to MODEL_LAGS.

    Returns:
        RegressionModel: Output and fitted model instance
    """
    model_lags = model_lags or MODEL_LAGS
    model_class = model_class or make_pipeline(LinearRegression(n_jobs=-1))
    linear_model = RegressionModel(
        lags=model_lags.num_lags,
        lags_past_covariates=model_lags.num_lags_past_covariates,
        lags_future_covariates=model_lags.num_lags_future_covariates,
        multi_models=False,
        output_chunk_length=1,
        model=model_class,
    )
    linear_model.fit(
        series=targets,
        past_covariates=past_features,
        future_covariates=future_features,
    )
    return linear_model

def fit_baseline_model(
    targets: list[TimeSeries], 
) -> NaiveSeasonal:
    """
    Initialise and fit a Darts model

    Args:
        targets (dict[int, TimeSeries]): Target timeseries

    Returns:
        baseline model: Output and fitted model instance
    """
    period = 32
    for m in range(12, 50):
        is_seasonal, period = check_seasonality(targets, m=m, max_lag=50, alpha=0.05)
        if is_seasonal:
            print(f"There is seasonality of order {period}.")
            break
    print(f"Fitting baseline model with period {period}.")
    baseline_model = NaiveSeasonal(K=period)  
    baseline_model.fit(
        series=targets,
    )
    return baseline_model