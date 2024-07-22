import logging
import traceback
from argparse import Namespace
from configs import UploadingConfig
from simple_parsing import ArgumentParser
from utils import config_logger, UploadScheduler


def get_args() -> Namespace:
    parser = ArgumentParser(
        description="Upload files to HuggingFace.",
    )
    parser.add_arguments(UploadingConfig, "config")
    return parser.parse_args()


def main(args: Namespace) -> None:
    config = args.config
    logging.info("Uploading files to HuggingFace Hub")

    if "*" in config.path.name or "." in config.path.name:
        folder_path = config.path.parent
        glob_pattern = config.path.name
    else:
        folder_path = config.path
        glob_pattern = None

    try:
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
        logging.info("Uploading files completed")
    except Exception:
        logging.info(f"Uploading files interrupted:\n{traceback.format_exc()}")


if __name__ == "__main__":
    args = get_args()
    config_logger(args.config.log_path)
    main(args=args)
