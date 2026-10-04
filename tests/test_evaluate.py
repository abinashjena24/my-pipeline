import numpy as np
import pytest

from src.evaluate import evaluate_classification


def test_evaluate_classification():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_proba = np.array([0.1, 0.6, 0.8, 0.9])
    
    metrics=evaluate_classification(y_true,y_pred,y_proba)
    
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(0.8)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    
def test_evaluate_classification_returns_all_metrics():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0])
    y_proba = np.array([0.1, 0.8, 0.2, 0.4])

    metrics = evaluate_classification(y_true, y_pred, y_proba)

    expected_metrics = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    }

    assert set(metrics.keys()) == expected_metrics