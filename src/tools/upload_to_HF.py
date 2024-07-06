import os
import sys

sys.path.append(os.getcwd())

import argparse
from logging import Logger
from src.utils import (
    get_default_arg_parser,
    get_logger,
    UploadScheduler,
)


def get_args() -> argparse.Namespace:
    parser = get_default_arg_parser(
        description="Upload files to HuggingFace repository",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repository ID",
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the file or directory to upload",
    )
    parser.add_argument(
        "--every",
        type=float,
        default=1,
        help="Upload every F minutes",
    )
    parser.add_argument(
        "--dir-in-repo",
        type=str,
        required=True,
        help="Path to upload in the repository",
    )
    parser.add_argument(
        "--repo-type",
        type=str,
        default="dataset",
        help="Type of the repository",
    )
    parser.add_argument(
        "--delete-after-upload",
        action="store_true",
        help="Delete the file or directory after uploading",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Zip the directory before uploading",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Reverse the sorted uploading list",
    )
    return parser.parse_args()


def main(args: argparse.Namespace, logger: Logger) -> None:
    logger.info("Uploading files to HuggingFace Hub")

    if "*" in os.path.basename(args.path):
        folder_path = os.path.dirname(args.path)
        glob_pattern = os.path.basename(args.path)
    elif "." in os.path.basename(args.path):
        folder_path = os.path.dirname(args.path)
        glob_pattern = os.path.basename(args.path)
    else:
        folder_path = args.path
        glob_pattern = None

    try:
        upload_scheduler = UploadScheduler(
            repo_id=args.repo_id,
            folder_path=folder_path,
            every=args.every,
            path_in_repo=args.dir_in_repo,
            repo_type=args.repo_type,
            logger=logger,
            glob_pattern=glob_pattern,
            delete_after_upload=args.delete_after_upload,
            overwrite=args.overwrite,
            zip=args.zip,
            reverse=args.reverse,
        )
        while True:
            if upload_scheduler.is_done:
                upload_scheduler.stop()
                break
        logger.info("Uploading files completed")
    except Exception as e:
        print(e)
        logger.info("Uploading files interrupted")


if __name__ == "__main__":
    args = get_args()
    logger = get_logger(name="upload_to_hf", log_path=args.log_path)
    main(args=args, logger=logger)
