import logging
from typing import Any

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.evaluate import evaluate_classification
from src.pipeline import build_estimator, build_preprocessing

logger = logging.getLogger(__name__)

MODEL_NAMES = ("random_forest", "xgboost", "lightgbm")
METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "roc_auc")

def run_benchmark(
    X: pd.DataFrame,
    y: pd.Series,
    model_params: dict[str, dict[str, Any]],
    cv_folds: int = 5,
    random_seed: int = 42,
) -> pd.DataFrame:
    
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows.")

    if X.empty:
        raise ValueError("Cannot benchmark an empty dataset.")

    if y.nunique() != 2:
        raise ValueError("Benchmark requires a binary target.")

    if y.value_counts().min() < cv_folds:
        raise ValueError(
            "Each target class must contain at least cv_folds samples."
        )

    numerical = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    
    cv = StratifiedKFold(n_splits=cv_folds,shuffle=True,random_state=random_seed,)
    
    folds = list(cv.split(X,y))
    records: list[dict[str, Any]] = []
    
    for model_name in MODEL_NAMES:
        if model_name not in MODEL_NAMES:
            raise ValueError(f"Missing parameters for {model_name}.")
        preprocessor = build_preprocessing(numerical,categorical)
        estimator = build_estimator(model_name,model_params[model_name])
        
        pipeline = Pipeline(
            steps=[("preprocessor",preprocessor),("model",estimator),]
        )
        
        fold_metrics: list[dict[str,float]] = []
        for fold_number, (train_idx, val_idx) in enumerate(folds, start=1):
            fold_pipeline = clone(pipeline)
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            fold_pipeline.fit(X_train, y_train)
            predictions = fold_pipeline.predict(X_val)
            probabilities = fold_pipeline.predict_proba(X_val)[:, 1]
            
            metrics = evaluate_classification(y_true=y_val,
                    y_pred=predictions,
                    y_proba=probabilities)
            fold_metrics.append(metrics)
            logger.info(
                "%s fold %d: ROC-AUC=%.4f",
                model_name,
                fold_number,
                metrics["roc_auc"],
            )

        metrics_df = pd.DataFrame(fold_metrics)

        record: dict[str, Any] = {"model": model_name}
        for metric in METRIC_NAMES:
            record[f"{metric}_mean"] = float(metrics_df[metric].mean())
            record[f"{metric}_std"] = float(metrics_df[metric].std(ddof=1))

        records.append(record)

    results = pd.DataFrame(records)
    return results.sort_values("roc_auc_mean", ascending=False).reset_index(
        drop=True
    )



    
    
