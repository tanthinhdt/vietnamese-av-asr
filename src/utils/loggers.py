from loguru import logger
from pathlib import Path
from typing import Union


def config_logger(log_file: Union[Path, str] = None) -> None:
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_file)
