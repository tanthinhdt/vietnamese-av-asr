import logging
import traceback
from loguru import logger
from argparse import Namespace
from configs import UploadingConfig
from simple_parsing import ArgumentParser
from utils import config_logger, UploadScheduler


logging.root.setLevel(logging.WARNING)


def get_args() -> Namespace:
    parser = ArgumentParser(
        description="Upload files to HuggingFace.",
    )
    parser.add_arguments(UploadingConfig, "config")
    return parser.parse_args()


def main(args: Namespace) -> None:
    config = args.config
    logger.info("Uploading files to HuggingFace Hub")

    if "*" in config.path.name or "." in config.path.name:
        folder_path = config.path.parent
        glob_pattern = config.path.name
    else:
        folder_path = config.path
        glob_pattern = None

    upload_scheduler = UploadScheduler(
        repo_id=config.repo_id,
        folder_path=folder_path,
        every=config.every_minutes,
        path_in_repo=config.dir_in_repo,
        repo_type=config.repo_type,
        glob_pattern=glob_pattern,
        delete_after_upload=config.delete_after_upload,
        overwrite=config.overwrite,
        zip=config.zip,
        reverse=config.reverse,
    )
    while True:
        if upload_scheduler.is_done:
            upload_scheduler.stop()
            break
    logger.info("Uploading files completed")


if __name__ == "__main__":
    args = get_args()
    config_logger(args.config.log_path)

    try:
        main(args=args)
    except Exception:
        logger.info(f"Uploading files interrupted:\n{traceback.format_exc()}")
