import os
import sys

sys.path.append(os.getcwd())

import argparse
from logging import Logger
from src.utils import (
    get_logger,
    zip_dir,
    upload_to_hf,
    exist_in_hf,
    get_default_arg_parser,
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
        "--delete-zip-after-upload",
        action="store_true",
        help="Delete the zip file after uploading",
    )
    return parser.parse_args()


def main(args: argparse.Namespace, logger: Logger) -> None:
    logger.info("Uploading files to HuggingFace Hub")
    if os.path.isdir(args.path) and args.zip:
        zip_dir(args.path, args.path)
        src_path = args.path + ".zip"
    else:
        src_path = args.path
    dest_path = os.path.join(args.dir_in_repo, os.path.basename(src_path))

    if args.overwrite or not exist_in_hf(
        repo_id=args.repo_id,
        path_in_repo=dest_path,
        repo_type=args.repo_type,
    ):
        upload_to_hf(
            src_path=src_path,
            dest_path=dest_path,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            logger=logger,
        )
    else:
        logger.warning(f"{dest_path} already exists in {args.repo_id} repository")
    logger.info("Uploading files completed")


if __name__ == "__main__":
    args = get_args()
    logger = get_logger(name="upload_to_hf", log_path=args.log_path)
    main(args=args, logger=logger)
