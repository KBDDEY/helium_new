__version__ = "1.0.0"

from helium2 import envs

__all__ = ["envs"]


# Suppress interfering loggers
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
