import os
import subprocess
from typing import Any

from src.models.taskers.tasker import Tasker
from src.models.taskers.checker import Checker
from src.models.utils import get_logger


class Normalizer(Tasker):
    """This class used to normalize frame rate , sample rate of video"""

    _FRAME_RATE: int = 25
    _SAMPLE_RATE: int = 16000
    _EXTENSION: str = '.mp4'
    _INTERIM_DIR: str = os.path.join(os.getcwd(), 'data', 'interim')

    def __init__(self, *args, **kwargs):
        super().__init__()

        self._FRAME_RATE = kwargs.get('sample_rate', self._FRAME_RATE)
        self._SAMPLE_RATE = kwargs.get('frame_rate', self._SAMPLE_RATE)
        self._EXTENSION = kwargs.get('extension', self._EXTENSION)
        self._INTERIM_DIR = kwargs.get('interim_dir', self._INTERIM_DIR)
        self._logger = get_logger(
            name=__name__,
            is_stream=True,
            log_path=None,
        )

    def do(self, metadata_dict: dict,  *args, **kwargs) -> dict:

        #metadata_dict['video_path'] = self._normalize(video_path=metadata_dict['video_path'])

        re_checker = Checker(
            frame_rate=self._FRAME_RATE,
            sample_rate=self._SAMPLE_RATE,
            extension=self._EXTENSION,
        )
        metadata_dict = re_checker.do(metadata_dict['video_path'])

        return metadata_dict

    def _normalize(self, video_path: str):
        prefix, file = os.path.split(p=video_path)
        file_name, ext = os.path.splitext(file)
        _tar_path = os.path.join(
            self._INTERIM_DIR, 'n_' + file_name + self._EXTENSION
        )
        _cmd = [
            'ffmpeg',
            '-y',
            '-i', '%s' % video_path,
            '-c:v', 'copy',
            '-c:a', 'copy',
            '-qscale:a', '0',
            '-r', '%d' % self._FRAME_RATE,
            '-ar', '%d' % self._SAMPLE_RATE,
            '-threads', '10',
            '-async', '1',
            '-f', self._EXTENSION[1:],
            '%s' % _tar_path,
            '-loglevel', 'panic',
        ]

        subprocess.run(
            args=_cmd,
            shell=False,
            stdout=None,
            stderr=None,
            capture_output=False,
            cwd=os.getcwd(),
        )

        if not os.path.isfile(_tar_path):
            self._logger.exception(f"Normalize sample rate video in path '{video_path}' fail.'")
            exit(123)
        return _tar_path




