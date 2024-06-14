import os
import sys
sys.path.append(os.getcwd())

import argparse

from src.data.utils import pipe, pipe_file, reset_demo, get_logger
from src.data.processors.tracker import track_video_file

logger = get_logger(name=__name__,is_stream=True)


def prepare_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Full pipeline to process end-to-end video (url/file path).")

    parser.add_argument(
        '--url',
        type=str,
        required=True,
        help='Url to process.'
    )

    parser.add_argument(
        '--file',
        type=str,
        required=False,
        default=None,
        help='Path of video file to process.'
    )

    parser.add_argument(
        '--do-file',
        required=False,
        default=False,
        action='store_true',
        help='Indicate process video file.'
    )

    parser.add_argument(
        '--channel-name',
        type=str,
        required=False,
        default='batch_88888',
        help='Channel name contain url/file.'
    )

    parser.add_argument(
        '--channel-volume',
        type=int,
        required=False,
        default=1,
        help="Number of urls for each channel."
    )

    parser.add_argument(
        '--tasks',
        type=str,
        metavar='TASKS',
        action='store',
        nargs='*',
        required=False,
        default=['full'],
        help='Tasks of pipeline (must consecutive). Order the tasks track ->\
              download -> asd -> crop -> vndetect -> transcribe. "full" execute all tasks.'
    )

    parser.add_argument(
        '--cache-dir',
        type=str,
        required=False,
        default=os.path.join(os.getcwd(), 'data', 'external'),
        help='Cache dir contain downloaded data.',
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        required=False,
        default=os.path.join(os.getcwd(), 'data', 'processed'),
        help='Directory contains all processed data.',
    )

    parser.add_argument(
        '--demo',
        required=False,
        default=False,
        action='store_true',
        help='Demo or collection'
    )

    parser.add_argument(
        '--overwrite',
        required=False,
        default=False,
        action='store_true',
        help='Overwrite existing'
    )

    parser.add_argument(
        '--clean-input',
        required=False,
        default=False,
        action='store_true',
        help="Clean input"
    )

    parser.add_argument(
        '--clean-output',
        required=False,
        default=False,
        action='store_true',
        help="Clean output"
    )

    return parser.parse_args()


def main(args: argparse.Namespace):
    if args.demo:
        args.overwrite = True
        args.channel_name = 'batch_88888'
        reset_demo(channel_name=args.channel_name)
    if args.do_file:
        # Only for demo
        if not args.demo:
            logger.warning("Do file only when demo")
        elif args.file is None:
            logger.warning('Missing \'--file\'')
        elif not os.path.isfile(args.file):
            logger.warning(f"No such file '{args.file}'")
        else:
            track_video_file(args.file, channel_name=args.channel_name, demo=args.demo)
            pipe_file(args=args)
    else:
        pipe(args=args)
    logger.info("Pipeline is finished. Load dataset from 'GSU24AI03-SU24AI21/vietnamese-av', with name 'batch_88888' to get processed data.")


if __name__ == '__main__':
    p_args = prepare_args()
    main(args=p_args)
