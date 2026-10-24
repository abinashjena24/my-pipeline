import numpy as np
import pandas as pd
import pytest
from scipy.sparse import issparse

from src.features import IQROutlierCapper
from src.pipeline import build_preprocessing


def test_iqr_capper_caps_values_properly() -> None:
    df_train = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]})
    df_test = pd.DataFrame({"val": [1000.0]})
    capper = IQROutlierCapper(columns=["val"], factor=1.5)
    capper.fit(df_train)
    assert capper.transform(df_test)["val"].iloc[0] < 1000.0


def test_iqr_capper_leakage_safety() -> None:
    df_train = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]})
    df_test = pd.DataFrame({"val": [1000.0]})
    capper = IQROutlierCapper(columns=["val"], factor=1.5)
    capper.fit(df_train)
    df_test_transformed = capper.transform(df_test)
    train_upper_bound = capper.caps_["val"][1]
    assert df_test_transformed["val"].iloc[0] == train_upper_bound


def test_iqr_capper_preserves_shape_and_columns() -> None:
    df = pd.DataFrame(
        {
            "amount": [1.0, 2.0, 3.0, 4.0, 5.0],
            "count": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
    capper = IQROutlierCapper().fit(df)

    transformed = capper.transform(df)

    assert transformed.shape == df.shape
    assert transformed.columns.tolist() == df.columns.tolist()
    assert transformed.index.equals(df.index)


def test_iqr_capper_rejects_missing_fitted_column() -> None:
    train = pd.DataFrame({"amount": [1.0, 2.0, 3.0, 4.0, 5.0]})
    capper = IQROutlierCapper(columns=["amount"]).fit(train)

    with pytest.raises(KeyError):
        capper.transform(pd.DataFrame({"other": [10.0]}))


def test_preprocessor_handles_unseen_categories() -> None:
    train = pd.DataFrame(
        {
            "amount": [1.0, 2.0, 3.0, 4.0],
            "category": ["A", "B", "A", "B"],
        }
    )
    preprocessor = build_preprocessing(["amount"], ["category"])
    preprocessor.fit(train)

    test = pd.DataFrame({"amount": [5.0, 6.0], "category": ["C", "A"]})
    transformed = preprocessor.transform(test)

    assert transformed.shape[0] == len(test)
    assert transformed.shape[1] > 1
    dense_transformed = transformed.toarray() if issparse(transformed) else transformed

    assert np.isfinite(dense_transformed).all()
