import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from configs.config import load_config
from src.evaluate import evaluate_classification
from src.logger import setup_logger
from src.pipeline import build_estimator, build_preprocessing
from src.tune import run_tuning

logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tabu lar ML model trainig pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default=r"C:\my things\my learning\My-pipeline\configs\base-config.yaml",
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Override the random seed"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Override the dataset path from the YAML configuration",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )
    parser.add_argument("--tune",
                        action="store_true",
                        help="Run Optuna hyperparameter optimization",
                        )

    return parser.parse_args()


def run_pipeline(
    config_path: str, seed: int | None = None, data_path: Path | None = None
) -> None:
    logger.info("Initiating ML training pipeline")
    logger.debug(f"Configuration path received: {config_path}")
    try:
        app_config = load_config(config_path)
        if seed is not None:
            app_config.project.random_seed = seed
            app_config.model.params["random_state"] = seed
        if data_path is not None:
            app_config.data.raw_data_path = str(data_path)
        logger.info(
            f"Loaded project:{app_config.project.name}(v{app_config.project.version})"
        )
        logger.info(f"Execution seed: {app_config.project.random_seed}")

        logger.info(f"Target model: {app_config.model.name}")
        data_file = Path(app_config.data.raw_data_path)

        if not data_file.is_file():
            raise FileNotFoundError(f"Dataset not found: {data_file}")

        logger.info(f"Data source: {data_file}")
        df=pd.read_csv(data_file)
        logger.info(f"Loaded dataset shape: {df.shape}")
        target_column = app_config.data.target_column
        X = df.drop(columns=[target_column])
        y = df[target_column]

        logger.info(f"Feature columns: {list(X.columns)}")
        numerical_features = X.select_dtypes(include="number").columns.tolist()
        categorical_features = X.select_dtypes(exclude="number").columns.tolist()

        logger.info(f"Numerical features: {numerical_features}")
        logger.info(f"Categorical features: {categorical_features}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=app_config.data.test_size,
            random_state=app_config.project.random_seed,
            stratify=y,
        )

        logger.info(
            f"Train shape: {X_train.shape}, Test shape: {X_test.shape}"
        )

        preprocessor=build_preprocessing(numerical_features=numerical_features,categorical_features=categorical_features,)
        logger.info("Preprocessing pipeline created successfully.")
        
        model = build_estimator(model_name=app_config.model.name,
                                model_params=app_config.model.params,
                                )
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        
        cv= StratifiedKFold(n_splits=5,
                            shuffle=True,
                            random_state=app_config.project.random_seed,
        )
        
        fold_metrics: list[dict[str, float]] = []
        
        logger.info("Initializng 5-fold stratified cross-validation.")
        for fold,(train_idx, val_idx) in enumerate(cv.split(X_train,y_train),start=1):
            
            X_fold_train = X_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]

            y_fold_train = y_train.iloc[train_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            fold_pipeline=clone(pipeline)
            fold_pipeline.fit(X_fold_train, y_fold_train)
            
            y_pred=fold_pipeline.predict(X_fold_val)
            
            y_proba = fold_pipeline.predict_proba(X_fold_val)[:,1]
            
            logger.info(f"Fold {fold} y_true: {y_fold_val.tolist()}")
            logger.info(f"Fold {fold} y_proba: {y_proba.tolist()}")
            
            metrics = evaluate_classification(
                y_true=y_fold_val,
                y_pred=y_pred,
                y_proba=y_proba,
            )
            fold_metrics.append(metrics)
            logger.info(
                f"Fold {fold}: "
                f"accuracy={metrics['accuracy']:.4f}, "
                f"precision={metrics['precision']:.4f}, "
                f"recall={metrics['recall']:.4f}, "
                f"f1={metrics['f1']:.4f}, "
                f"roc_auc={metrics['roc_auc']:.4f}"
            )
            
        metrics_df = pd.DataFrame(fold_metrics)
            
        mean_metrics = metrics_df.mean()
            
        logger.info("Cross-validation completed successfully.")

        logger.info(
            f"Mean Accuracy: {mean_metrics['accuracy']:.4f}"
        )

        logger.info(
            f"Mean Precision: {mean_metrics['precision']:.4f}"
        )

        logger.info(
            f"Mean Recall: {mean_metrics['recall']:.4f}"
        )

        logger.info(
            f"Mean F1: {mean_metrics['f1']:.4f}"
        )

        logger.info(
            f"Mean ROC-AUC: {mean_metrics['roc_auc']:.4f}"
        )

        
        pipeline.fit(X_train,y_train)
        logger.info("Model training completed successfully.")
        y_test_pred = pipeline.predict(X_test)
        y_test_proba = pipeline.predict_proba(X_test)[:, 1]

        test_metrics = evaluate_classification(
            y_true=y_test,
            y_pred=y_test_pred,
            y_proba=y_test_proba,
        )

        logger.info(
                f"Test metrics: "
                f"accuracy={test_metrics['accuracy']:.4f}, "
                f"precision={test_metrics['precision']:.4f}, "
                f"recall={test_metrics['recall']:.4f}, "
                f"f1={test_metrics['f1']:.4f}, "
                f"roc_auc={test_metrics['roc_auc']:.4f}"
        )
        
        logger.info(f"Target column: {app_config.data.target_column}")

        logger.info(f"Test size: {app_config.data.test_size}")

        logger.info(
            "Model hyperparameters: %s",
            app_config.model.params,
        )
        logger.info("Configuration validation and pipeline assembly succeeded.")
    except Exception:
        logger.exception("Fatal error encountered during pipeline startup")
        raise


def main() -> None:
    args = parse_args()

    setup_logger(args.log_level)

    global logger
    logger = logging.getLogger(__name__)

    if args.tune:
        app_config = load_config(args.config)

        if args.seed is not None:
            app_config.project.random_seed = args.seed

        if args.data_path is not None:
            app_config.data.raw_data_path = str(args.data_path)

        data_file = Path(app_config.data.raw_data_path)

        if not data_file.is_file():
            raise FileNotFoundError(f"Dataset not found: {data_file}")

        logger.info("Starting Optuna hyperparameter tuning.")
        logger.info("Data source: %s", data_file)
        logger.info("Target model: %s", app_config.model.name)

        df = pd.read_csv(data_file)

        target_column = app_config.data.target_column
        X = df.drop(columns=[target_column])
        y = df[target_column]

        search_space = {
            name: config.model_dump()
            for name, config in app_config.tuning.search_space.items()
        }

        study = run_tuning(
            X=X,
            y=y,
            model_name=app_config.model.name,
            base_params=app_config.model.params,
            search_space=search_space,
            n_trials=app_config.tuning.n_trials,
            cv_folds=app_config.tuning.cv_folds,
            random_seed=app_config.project.random_seed,
        )

        logger.info("Best ROC-AUC: %.4f", study.best_value)
        logger.info("Best parameters: %s", study.best_params)

        return

    run_pipeline(
        config_path=args.config,
        seed=args.seed,
        data_path=args.data_path,
    )


if __name__ == "__main__":
    main()
