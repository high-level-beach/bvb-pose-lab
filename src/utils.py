import logging
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "pose_lab.log"


def get_logger(name: str) -> logging.Logger:
    """
    Creates and returns a logger instance scoped to the given module name.

    The logger:
    - logs INFO and above to console
    - logs DEBUG and above to a rotating file
    - uses the format: [LEVEL] [module] message

    Parameters
    ----------
    name : str
        Name of the logger (typically __name__ of the calling module)

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already initialized

    logger.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter(
        "[%(levelname)s] [%(name)s.%(funcName)s:%(lineno)d] %(message)s"
    )

    # Console handler (INFO+)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler (DEBUG+)
    fh = logging.FileHandler(LOG_FILE)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
