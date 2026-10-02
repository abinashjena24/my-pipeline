import pandas as pd

from src.pipeline import build_preprocessing


def test_preprocessing_pipeline() -> None:
    data = pd.DataFrame(
        {
            "age": [20, 25, 30, 35, 40],
            "salary": [20000, 30000, 40000, 50000, 60000],
            "city": ["Delhi", "Mumbai", "Delhi", "Pune", "Mumbai"],
        }
    )

    numerical_features = ["age", "salary"]
    categorical_features = ["city"]

    preprocessor = build_preprocessing(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
    )

    transformed = preprocessor.fit_transform(data)

    assert transformed.shape[0] == len(data)
    assert transformed.shape[1] > len(numerical_features)
    
def test_preprocessing_pipeline_leakage_safety() -> None:
        train_data = pd.DataFrame(
        {
            "age": [20, 21, 22, 23, 24],
            "salary": [20000, 21000, 22000, 23000, 24000],
            "city": ["Delhi", "Mumbai", "Delhi", "Pune", "Mumbai"],
        }
    )

        test_data = pd.DataFrame(
        {
            "age": [1000],
            "salary": [1_000_000],
            "city": ["Bangalore"],
        }
    )

        preprocessor = build_preprocessing(
        numerical_features=["age", "salary"],
        categorical_features=["city"],
    )

        preprocessor.fit(train_data)

        transformed_test = preprocessor.transform(test_data)

        assert transformed_test.shape[0] == 1