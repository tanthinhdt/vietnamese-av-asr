import os
import argparse
from glob import glob
from logging import Logger
from datasets import Dataset
from src.utils import get_default_arg_parser, get_logger, extract_zip


def get_args() -> argparse.Namespace:
    parser = get_default_arg_parser(description="Extract zip files")
    parser.add_argument(
        "--dir-or-zip-file",
        type=str,
        required=True,
        help="Path to the directory contains zip files or zip file.",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=None,
        help="Number of processes to use.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--delete-after-extract",
        action="store_true",
        help="Delete zip files after extracting.",
    )
    return parser.parse_args()


def extract(
    sample: dict,
    output_dir: str,
    delete_after_extract: bool,
    logger: Logger,
) -> dict:
    extract_zip(sample["file"], output_dir, logger, delete_after_extract)
    return sample


def main(args: argparse.Namespace, logger: Logger) -> None:
    logger.info("Extracting zip files")
    if not os.path.exists(args.dir_or_zip_file):
        logger.error(f"{args.dir_or_zip_file} does not exist")
        return
    elif os.path.isfile(args.dir_or_zip_file):
        file_paths = [args.dir_or_zip_file]
        if args.output_dir is None:
            output_dir = os.path.dirname(args.dir_or_zip_file)
    else:
        file_paths = glob(os.path.join(args.dir_or_zip_file, "*.zip"))
        if args.output_dir is None:
            output_dir = args.dir_or_zip_file
    os.makedirs(output_dir, exist_ok=True)

    dataset = Dataset.from_dict({"file": file_paths})
    logger.info(f"Found {len(dataset)} zip file(s)")
    dataset.map(
        extract,
        fn_kwargs={
            "output_dir": output_dir,
            "delete_after_extract": args.delete_after_extract,
            "logger": logger,
        },
        num_proc=args.num_proc,
    )
    logger.info("Extraction completed")


if __name__ == "__main__":
    args = get_args()
    logger = get_logger(name="extract_zip", log_path=args.log_path)
    main(args=args, logger=logger)
