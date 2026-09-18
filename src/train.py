import argparse
import sys
import logging
from pathlib import Path


#to get the path even if it is inside another folder of the same file 
ROOT_DIR=Path(__file__).resolve().parent.parent
if ROOT_DIR not in sys.path:
    sys.path.append(str(ROOT_DIR))
    
from src.logger import setup_logger
from configs.config import load_config

setup_logger()
logger=logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser=argparse.ArgumentParser(description="Tabular ML model trainig pipeline")
    parser.add_argument("--config",type=str,default=r"C:\my things\my learning\My-pipeline\configs\base-config.yaml",help="Path to the YAML configuration file")
    parser.add_argument("--seed",type=int,default=None,help="Override the random seed")
    parser.add_argument("--data-path",type=Path,default=None,help="Override the dataset path from the YAML configuration")
    parser.add_argument("--log-level",type=str,choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"],default="INFO",help="Set the logging level")
    
    return parser.parse_args()

def run_pipeline(config_path:str,seed:int|None=None,data_path:Path|None=None)->None:
    logger.info("Initiating ML training pipeline")
    try:
        app_config=load_config(config_path)
        if seed is not None:
            app_config.project.random_seed=seed
            app_config.model.random_state=seed
        if data_path is not None:
            app_config.data.raw_data_path=str(data_path)
        logger.info(f"Loaded project:{app_config.project.name}"f"(v{app_config.project.version})")
        logger.info(
            f"Execution seed: {app_config.project.random_seed}"
        )

        logger.info(
            f"Target model: {app_config.model.name}"
        )

        logger.info(
            f"Data source: {app_config.data.raw_data_path}"
        )

        logger.info(
            f"Target column: {app_config.data.target_column}"
        )

        logger.info(
            f"Test size: {app_config.data.test_size}"
        )

        logger.info(
            f"Model hyperparameters: "
            f"n_estimators={app_config.model.n_estimators}, "
            f"max_depth={app_config.model.max_depth}, "
            f"random_state={app_config.model.random_state}"
        )

        logger.info(
            "Configuration validation and pipeline assembly succeeded."
        )
    except Exception:
        logger.exception("Fatal error encountered during pipeline startup")
        raise

def main()->None:
    args=parse_args()
    run_pipeline(config_path=args.config,seed=args.seed,data_path=args.data_path)
if __name__ == "__main__":
    main()
