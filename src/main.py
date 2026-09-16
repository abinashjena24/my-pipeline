import logging
from database import connect_database
from logger import setup_logger

setup_logger()

logger = logging.getLogger(__name__)

logger.info("Application started")

connect_database()

logger.info("Application finished")