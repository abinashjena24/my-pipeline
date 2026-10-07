from typing import Any

import yaml
from pydantic import BaseModel, Field, StrictInt, ValidationError


class ProjectConfig(BaseModel):
    name: str
    version: str
    random_seed: int


class DataConfig(BaseModel):
    raw_data_path: str
    target_column: str
    test_size: float = Field(gt=0, lt=1)


class ModelConfig(BaseModel):
    name: str
    params: dict[str, Any]

class SearchSpaceConfig(BaseModel):
    type: str
    low: float
    high: float
    log: bool = False


class TuningConfig(BaseModel):
    n_trials: StrictInt = Field(gt=0)
    cv_folds: StrictInt = Field(gt=1)
    metric: str
    search_space: dict[str, SearchSpaceConfig]
class AppConfig(BaseModel):
    project: ProjectConfig
    data: DataConfig
    model: ModelConfig
    tuning: TuningConfig


def load_config(config_path: str = "configs/base-config.yaml") -> AppConfig:
    try:
        with open(config_path) as file:
            data = yaml.safe_load(file)
        if data is None:
            raise ValueError(f"Configuration file is empty: {config_path}")
        config = AppConfig(**data)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"confugaration file not found:{config_path}")
    except ValidationError as e:
        raise ValueError(f"Invalid configuration: {e}") from e
    except yaml.YAMLError:
        raise ValueError(f"Invalid YAML syntax in configuration file:{config_path}")


if __name__ == "__main__":
    # by importing this file the print statements would have run automatically #
    # so we use this if __name__ =" __main__"

    config = load_config()

    print("Project:", config.project.name)
    print("Random Seed:", config.project.random_seed)

    print("Data Path:", config.data.raw_data_path)
    print("Target:", config.data.target_column)
    print("Test Size:", config.data.test_size)

    print("Model:", config.model.name)
    print("Model Parameters:", config.model.params)         
