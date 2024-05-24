import os
import sys

sys.path.append(os.getcwd())

import argparse
from logging import Logger
from src.utils import get_logger, download_from_hf, get_default_arg_parser


def get_args():
    parser = get_default_arg_parser(
        description="Download files from HuggingFace repository",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repository ID",
    )
    parser.add_argument(
        "--path-in-repo",
        type=str,
        required=True,
        help="Path to download in the repository",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files",
    )
    parser.add_argument(
        "--repo-type",
        type=str,
        default="dataset",
        help="Type of the repository",
    )
    return parser.parse_args()


def main(args: argparse.Namespace, logger: Logger):
    """
    Main function to download files from HuggingFace repository.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments.
    """
    logger.info("Downloading files from HuggingFace Hub")
    logger.info("If your data is private, please login to Hugging or set it public")
    logger.info(f"Write mode: {'overwrite' if args.overwrite else 'skip'}")
    download_from_hf(
        repo_id=args.repo_id,
        path_in_repo=args.path_in_repo,
        output_dir=os.path.normpath(args.output_dir),
        logger=logger,
        overwrite=args.overwrite,
        repo_type=args.repo_type,
    )
    logger.info("Download completed")


if __name__ == "__main__":
    args = get_args()
    logger = get_logger(name="download_from_hf", log_path=args.log_path)
    main(args=args, logger=logger)
