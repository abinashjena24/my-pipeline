import logging
logger=logging.getLogger(__name__)
def connect_database():
    logger.info("Connecting to database")
    logger.debug("Checking database configuration")
    try:
        result=10/0
    except Exception:
        logger.exception("Database connection failed")
        