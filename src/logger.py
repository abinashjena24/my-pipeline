import logging
from pathlib import Path


def setup_logger(log_level:str="INFO"):
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging,log_level.upper()))

    if not root_logger.handlers:
        log_path = Path("app.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # File handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)

        # Attach handlers to root logger
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)