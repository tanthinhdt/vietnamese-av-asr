import os
import argparse
from glob import glob
from logging import Logger
from utils import get_default_arg_parser, get_logger, zip_dir


def get_args() -> argparse.Namespace:
    parser = get_default_arg_parser(
        description="Zip directories",
    )
    parser.add_argument(
        "--dir-path",
        type=str,
        required=True,
        help="Path to the directory to zip",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to save the zip file",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the existing zip file",
    )
    parser.add_argument(
        "--delete-after-zip",
        action="store_true",
        help="Delete the directory after zipping",
    )
    return parser.parse_args()


def main(args: argparse.Namespace, logger: Logger) -> None:
    logger.info("Zipping directories")
    if os.path.isdir(args.dir_path):
        dir_paths = [args.dir_path]
    elif args.dir_path.endswith("/*"):
        dir_paths = glob(args.dir_path)
    else:
        logger.error(f"{args.dir_path} is not a directory or a glob pattern")
        return

    logger.info(f"Found {len(dir_paths)} directories to zip")
    for i, dir_path in enumerate(dir_paths):
        logger.info(f"[{i + 1}/{len(dir_paths)}] Zipping directory: {dir_path}")
        zip_dir(
            dir_path=dir_path,
            output_dir=args.output_dir,
            logger=logger,
            overwrite=args.overwrite,
            delete_after_zip=args.delete_after_zip,
        )
    logger.info("Finished zipping directories")


if __name__ == "__main__":
    args = get_args()
    logger = get_logger(name="zip_dirs")
    main(args=args, logger=logger)
