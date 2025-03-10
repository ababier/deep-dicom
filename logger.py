import logging
import logging.config


def setup_logging(log_file: str = None, level: int = logging.INFO) -> None:
    """
    Set up logging configuration.

    Args:
        log_file (str): Optional file path for log output. If provided, a rotating file handler will be added.
        level (int): Logging level (e.g., logging.DEBUG, logging.INFO, etc.)
    """
    # Base configuration using a dict
    config = {
        "version": 1,
        "disable_existing_loggers": False,  # Keep existing loggers if any
        "formatters": {
            "detailed": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s", "datefmt": "%Y-%m-%d %H:%M:%S"},
        },
        "handlers": {
            # Console handler configuration
            "console": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "detailed",
                "stream": "ext://sys.stdout",
            },
        },
        "root": {
            "handlers": ["console"],
            "level": level,
        },
    }

    # If a log file is provided, add a rotating file handler
    if log_file:
        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": level,
            "formatter": "detailed",
            "filename": log_file,
            "maxBytes": 10 * 1024 * 1024,  # 10 MB per file
            "backupCount": 5,  # Keep up to 5 backup files
            "encoding": "utf8",
        }
        # Add the file handler to the root logger
        config["root"]["handlers"].append("file")

    # Apply the configuration
    logging.config.dictConfig(config)
