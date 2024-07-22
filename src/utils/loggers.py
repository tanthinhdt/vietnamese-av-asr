import sys
import logging
from pathlib import Path


def config_logger(log_file: str = None) -> None:    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_dir = Path(log_file).parent
        if not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(filename=log_file))
    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        format="[%(asctime)s] [%(filename)s:%(lineno)d] [%(levelname)s] - %(message)s", 
        handlers=handlers
    )
