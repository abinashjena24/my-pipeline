from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline


def save_pipeline(pipeline: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline,path)

def load_pipeline(path: Path) -> Pipeline:
    if not path.is_file():
        raise FileNotFoundError(f"Pipeline artifact not found: {path}")
    loaded: Any = joblib.load(path)

    if not isinstance(loaded, Pipeline):
        raise TypeError("The loaded artifact is not a scikit-learn Pipeline.")
    
    return loaded