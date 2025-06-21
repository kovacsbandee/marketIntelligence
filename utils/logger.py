from pathlib import Path
import logging


def get_logger(script_name: str) -> logging.Logger:
    """Set up a logger for management scripts.

    Creates or rotates log files so that only the latest and the previous
    run's log are kept.

    Args:
        script_name: Base name of the calling script without extension.

    Returns:
        Configured logger instance.
    """
    root_dir = Path(__file__).resolve().parents[1]
    log_dir = root_dir / "logs" / "management"
    log_dir.mkdir(parents=True, exist_ok=True)

    current_log = log_dir / f"{script_name}.log"
    previous_log = log_dir / f"{script_name}.prev.log"

    if current_log.exists():
        if previous_log.exists():
            previous_log.unlink()
        current_log.rename(previous_log)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(current_log, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    return logging.getLogger(script_name)