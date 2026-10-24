import pytest
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.pipeline import build_estimator


def test_build_random_forest() -> None:
    estimator = build_estimator(
        "random_forest",
        {"n_estimators": 10},
    )

    assert isinstance(estimator, RandomForestClassifier)
    assert estimator.get_params()["n_estimators"] == 10


def test_build_xgboost_with_class_imbalance() -> None:
    estimator = build_estimator(
        "xgboost",
        {
            "n_estimators": 10,
            "scale_pos_weight": 3.0,
        },
    )

    assert isinstance(estimator, XGBClassifier)
    assert estimator.n_estimators == 10
    assert estimator.scale_pos_weight == 3.0


def test_build_lightgbm_with_class_imbalance() -> None:
    estimator = build_estimator(
        "lightgbm",
        {
            "n_estimators": 10,
            "is_unbalance": True,
        },
    )

    assert isinstance(estimator, LGBMClassifier)
    assert estimator.n_estimators == 10
    assert estimator.get_params()["is_unbalance"] is True


def test_unsupported_model_raises_error() -> None:
    with pytest.raises(ValueError, match="Unsupported model"):
        build_estimator("invalid_model", {})
