import pandas as pd

from src.features import IQROutlierCapper


def test_iqr_capper_caps_values_properly() -> None:
    df_train = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]})
    df_test = pd.DataFrame({"val": [1000.0]})
    capper=IQROutlierCapper(columns=["val"],factor=1.5)
    capper.fit(df_train)
    assert capper.transform(df_test)["val"].iloc[0]<1000.0

def test_iqr_capper_leakage_safety() -> None:
    df_train = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]})
    df_test = pd.DataFrame({"val": [1000.0]})
    capper = IQROutlierCapper(columns=["val"], factor=1.5)
    capper.fit(df_train)
    df_test_transformed=capper.transform(df_test)
    train_upper_bound=capper.caps_["val"][1]
    assert df_test_transformed["val"].iloc[0]==train_upper_bound