from typing import Any, List

from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.features import IQROutlierCapper


def build_preprocessing(
    numerical_features: List[str], categorical_features: List[str]
) -> ColumnTransformer:
    numerical_pipeline = Pipeline(
        steps=[
            ("outlier_capper", IQROutlierCapper(columns=numerical_features)),
            ("impute", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, numerical_features),
            ("categorical", categorical_pipeline, categorical_features)
        ]
    )
    return preprocessor

def build_estimator(model_name: str,model_params: dict[str,Any]) -> Any:
    models={"random_forest": RandomForestClassifier,
           "xgboost": XGBClassifier,
           "lightgbm": LGBMClassifier,
           }
    
    if model_name not in models:
        raise ValueError(
            f"Unsupported model: {model_name}. "
            f"Supported models: {list(models)}"
        )
     
    return models[model_name](**model_params)
