import os
import sys

sys.path.append(os.getcwd())
import argparse

from src.data.processors.tracker import get_metadata, divide_metadata
from src.data.utils import get_logger

logger = get_logger(name=__name__, is_stream=True)


def prepare_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Prepare url to download video, url saved in form table in .parquet file.')

    parser.add_argument(
        '--url',
        type=str,
        required=True,
        help='Url to track, prepare to download.'
    )

    parser.add_argument(
        '--channel-name',
        type=str,
        required=False,
        default='batch_88888',
        help='Name of channel url belong to.'
    )

    parser.add_argument(
        '--channel-volume',
        type=int,
        required=False,
        default=-1,
        help='Number tracked url for each channel.'
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        required=False,
        default=False,
        help='Overwrite if file exists.'
    )

    parser.add_argument(
        '--demo',
        required=False,
        default=False,
        action='store_true',
        help='Indicate demo or collection'
    )

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    metadata = get_metadata(
        url=args.url,
    )
    if metadata:
        if args.demo:
            args.overwrite = True
        divide_metadata(
            metadata=metadata,
            volume=args.channel_volume,
            channel=args.channel_name,
            demo=args.demo,
            overwrite=args.overwrite,
        )


if __name__ == '__main__':
    p_args = prepare_args()
    main(args=p_args)