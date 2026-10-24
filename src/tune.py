import logging
from typing import Any

import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.pipeline import build_estimator, build_preprocessing

logger = logging.getLogger(__name__)


def suggest_parameters(
    trial: optuna.Trial, search_space: dict[str, dict[str, Any]]
) -> Any:
    params: dict[str, Any] = {}

    for name, config in search_space.items():
        parameter_type = config["type"]

        if parameter_type == "int":
            params[name] = trial.suggest_int(name, config["low"], config["high"])
        elif parameter_type == "float":
            params[name] = trial.suggest_float(
                name, config["low"], config["high"], log=config.get("log", False)
            )
        else:
            raise ValueError(
                f"Unsupported search-space parameter type: {parameter_type}"
            )
    return params


def create_pipeline(
    model_name: str,
    model_params: dict[str, Any],
    numerical_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    preprocessor = build_preprocessing(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
    )

    estimator = build_estimator(
        model_name=model_name,
        model_params=model_params,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )


def objective(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    base_params: dict[str, Any],
    search_space: dict[str, dict[str, Any]],
    cv_folds: int,
    random_seed: int,
) -> float:
    trial_params = suggest_parameters(trial, search_space)

    model_params = {**base_params, **trial_params}

    numerical_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)

    fold_scores: list[float] = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        X_train = X.iloc[train_idx]
        x_val = X.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        pipeline = create_pipeline(
            model_name=model_name,
            model_params=model_params,
            numerical_features=numerical_features,
            categorical_features=categorical_features,
        )
        pipeline = clone(pipeline)
        pipeline.fit(X_train, y_train)

        y_proba = pipeline.predict_proba(x_val)[:, 1]
        fold_score = float(roc_auc_score(y_val, y_proba))
        fold_scores.append(fold_score)

        trial.report(fold_score, step=fold)

        if trial.should_prune():
            raise optuna.TrialPruned()

    mean_score = sum(fold_scores) / len(fold_scores)

    logger.info(
        "Trial %d completed: mean ROC-AUC=%.4f",
        trial.number,
        mean_score,
    )

    return mean_score


def run_tuning(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    base_params: dict[str, Any],
    search_space: dict[str, dict[str, Any]],
    n_trials: int,
    cv_folds: int,
    random_seed: int,
) -> optuna.Study:

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_seed),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(
        lambda trial: objective(
            trial=trial,
            X=X,
            y=y,
            model_name=model_name,
            base_params=base_params,
            search_space=search_space,
            cv_folds=cv_folds,
            random_seed=random_seed,
        ),
        n_trials=n_trials,
    )

    logger.info("Optimization completed.")
    logger.info("Best ROC-AUC: %.4f", study.best_value)
    logger.info("Best parameters: %s", study.best_params)

    return study
