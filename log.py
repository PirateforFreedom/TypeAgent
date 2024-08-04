import logging
from pathlib import Path
from sys import stdout
from typing import Optional
from logging.handlers import RotatingFileHandler
from logging.config import dictConfig
from settings import settings
# from constants import (
#     LOGGER_DEFAULT_LEVEL,
#     LOGGER_DIR,
#     LOGGER_FILE_BACKUP_COUNT,
#     LOGGER_FILENAME,
#     LOGGER_MAX_FILE_SIZE,
#     LOGGER_NAME,
# )

# # Checking if log directory exists
# if not os.path.exists(LOGGER_DIR):
#     os.makedirs(LOGGER_DIR, exist_ok=True)
selected_log_level = logging.DEBUG if settings.debug else logging.INFO
# # Create logger for typeagent
# logger = logging.getLogger(LOGGER_NAME)
# logger.setLevel(LOGGER_DEFAULT_LEVEL)

# create console handler and set level to debug
# console_handler = logging.StreamHandler()
def _setup_logfile() -> "Path":
    """ensure the logger filepath is in place
    Returns: the logfile Path
    """
    logfile = Path(settings.typeagent_dir / "logs" / "TypeAgent.log")
    logfile.parent.mkdir(parents=True, exist_ok=True)
    logfile.touch(exist_ok=True)
    return logfile
# create rotatating file handler
# file_handler = RotatingFileHandler(
#     os.path.join(LOGGER_DIR, LOGGER_FILENAME), maxBytes=LOGGER_MAX_FILE_SIZE, backupCount=LOGGER_FILE_BACKUP_COUNT
# )

# # create formatters
# console_formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")  # not datetime
# file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# add formatter to console handler
# console_handler.setFormatter(console_formatter)

# add formatter for file handler
# file_handler.setFormatter(file_formatter)

# # add ch to logger
# logger.addHandler(console_handler)
# logger.addHandler(file_handler)
# TODO: production logging should be much less invasive
DEVELOPMENT_LOGGING = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {
        # "standard": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
         "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "no_datetime": {
            "format": "%(name)s - %(levelname)s - %(message)s",
        }
    },
    "handlers": {
        "console": {
            "level": selected_log_level,
            "class": "logging.StreamHandler",
            "stream": stdout,
            "formatter": "no_datetime",
        },
        "file": {
            "level": "DEBUG",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": _setup_logfile(),
            "maxBytes": 1024**2 * 10,
            "backupCount": 3,
            "formatter": "standard",
        },
    },
    "loggers": {
        "TypeAgent": {
            "level": logging.DEBUG if settings.debug else logging.INFO,
            "handlers": [
                "console",
                "file",
            ],
            "propagate": False,
        },
        "uvicorn": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False,
        },
    },
}
def get_logger(name: Optional[str] = None) -> "logging.Logger":
    """returns the project logger, scoped to a child name if provided
    Args:
        name: will define a child logger
    """
    dictConfig(DEVELOPMENT_LOGGING)
    # logging.config.dictConfig(DEVELOPMENT_LOGGING)
    parent_logger = logging.getLogger("TypeAgent")
    if name:
        return parent_logger.getChild(name)
    return parent_logger