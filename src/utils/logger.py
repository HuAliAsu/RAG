import logging
import sys
from logging.handlers import RotatingFileHandler

# These settings can later be loaded from config.ini
LOG_FILE_PATH = "logs/app.log" # Relative to RAG_Workbench directory
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s | %(levelname)s | %(module)s:%(funcName)s:%(lineno)d | %(message)s'
CONSOLE_LOG_FORMAT = '%(asctime)s | %(levelname)s | %(message)s'

# Get the root logger
logger = logging.getLogger('RAG_Workbench')
logger.setLevel(LOG_LEVEL)

# Prevent adding handlers multiple times if the module is reloaded
if not logger.handlers:
    # File handler with rotation
    # The path should be relative to where the app is run.
    # Assuming the app is run from the RAG_Workbench directory.
    file_handler = RotatingFileHandler(
        LOG_FILE_PATH,
        maxBytes=10 * 1024 * 1024,  # 10 MB from config.ini
        backupCount=5,              # 5 from config.ini
        encoding='utf-8'
    )
    file_formatter = logging.Formatter(LOG_FORMAT)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler for UI or terminal feedback
    stream_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(CONSOLE_LOG_FORMAT)
    stream_handler.setFormatter(console_formatter)
    logger.addHandler(stream_handler)

    logger.info("Logger has been initialized.")
else:
    logger.info("Logger already initialized.")