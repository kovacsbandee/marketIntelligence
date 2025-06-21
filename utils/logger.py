import logging
import sys
from pathlib import Path


def get_logger(name: str, log_dir: str = "logs/management", level: int = logging.INFO) -> logging.Logger:
    """Return a logger that writes to ``log_dir`` and stdout.

    Parameters
    ----------
    name:
        Name of the logger and output log file.
    log_dir:
        Directory where log files are stored.
    level:
        Logging level.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(Path(log_dir) / f"{name}.log", mode="w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
