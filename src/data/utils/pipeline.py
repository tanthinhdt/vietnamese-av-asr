import os
import subprocess

from src.data.utils import get_logger
from src.data.utils.demo import check_metadata_demo, reset_demo
from src.data.processors.tracker import track_video_file
import argparse

logger = get_logger(name=__name__, is_stream=True)

_TASK_ORDER = {
    'track':        0,
    'download':     1,
    'asd':          2,
    'crop':         3,
    'vndetect':     4,
    'transcribe':   5,
}


def pipe_url(args: argparse.Namespace) -> None:
    if 'track' not in args.tasks:
        logger.debug("'track' task is not in pipeline, so later tasks are executed base on channel name.")
    _last_task = None
    for task in args.tasks:
        if _last_task is not None:
            if _TASK_ORDER[task] - _TASK_ORDER[_last_task] != 1:
                logger.warning(f"Last task '{_last_task}' is not followed by current task '{task}'.")
        _last_task = task
        if task == 'track':
            task_cmd = [
                'python',
                'src/data/tasks/track.py',
                '--url',
                args.url,
                '--channel-name',
                args.channel_name,
            ]
            if args.demo:
                task_cmd.append('--demo')
        else:
            task_cmd = [
                'python',
                'src/data/tasks/process.py',
                '--task',
                task,
                '--channel-names',
                args.channel_name,
                '--output-dir',
                args.output_dir,
                '--cache-dir',
                args.cache_dir,
                '--upload-to-hub',
            ]
            if args.clean_input:
                task_cmd.append('--clean-input')
            if args.clean_output:
                task_cmd.append('--clean-output')
        if args.overwrite:
            task_cmd.append('--overwrite')

        res = subprocess.run(
            task_cmd,
            shell=False,
            stdout=None,
            capture_output=False,
            cwd=os.getcwd()
        )
        if res.returncode == 111 or not check_metadata_demo(task=task,channel_name=args.channel_name):
            logger.info(f"Fail process channel '{args.channel_name}' in task '{task}'. Program is exit.")
            exit(123)

def pipe_file(args: argparse.Namespace) -> None:
    track_video_file(args.file, channel_name=args.channel_name,demo=args.demo)
    if 'track' in args.tasks:
        logger.info(f'File pipe no need \'track\'.')
        args.tasks.remove('track')
    if 'download' in args.tasks:
        logger.info(f'File pipe no need \'download\'.')
        args.tasks.remove('download')

    pipe_url(args=args)


def pipe(args: argparse.Namespace) -> None:
    if 'full' in args.tasks:
        args.tasks = [
            'track',
            'download',
            'asd',
            'crop',
            'vndetect',
            'transcribe',
        ]
    pipe_url(args=args)