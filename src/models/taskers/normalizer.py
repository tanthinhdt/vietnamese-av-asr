import os
import subprocess
from typing import Any

from .tasker import Tasker
from .checker import Checker
from src.models.utils.dirs import _DATASET_DIR
from src.models.utils.logging import get_logger

logger = get_logger(
    name=__name__,
    log_path=None,
    is_stream=True
)


class Normalizer(Tasker):

    _STANDARD_V_CODECS: list = ['libx264', 'libaom-av1']
    _STANDARD_A_CODECS: list = ['aac', 'pcm_s16le']

    def __init__(self, v_codec: str = None, a_codec: str = None):
        super().__init__()
        self.v_codec = v_codec if v_codec else self._STANDARD_V_CODECS[0]
        self.a_codec = a_codec if a_codec else self._STANDARD_A_CODECS[0]
        self._checker = Checker()

    def do(self, metadata_dict: dict, *args, **kwargs) -> dict:
        video_path = metadata_dict['video_path']

        if metadata_dict['v_codec'] is not None and metadata_dict['v_codec'] == 'av1':
            video_path = self._normalize_av1_codec(video_path=video_path)

        return self._checker.do(video_path=video_path)

    def _normalize_av1_codec(self, video_path: str):
        _output_path = os.path.join(
            'data/raw', 'demo.mp4'
        )

        _cmd = [
            'ffmpeg', '-y',
            '-c:v', 'libaom-av1',
            '-i', video_path,
            '-map', '0:v:0',
            '-map', '0:a:0',
            '-c:v', self.v_codec,
            '-c:a', self.a_codec,
            '-f', 'avi',
            _output_path,
            '-loglevel', 'panic',
        ]

        subprocess.run(_cmd, shell=False, capture_output=False, stdout=None)

        if not os.path.isfile(_output_path):
            logger.warning(f"Normalize codec video '{video_path}' fail.")
            return video_path
        return _output_path

    # Add blank sound to video, in order to active speaker
    def _add_silent_audio(self, video_path: str):
        _output_path = os.path.join(
            'data/raw', 'demo.mp4'
        )
        _cmd = [
            'ffmpeg', '-y',
            '-f', 'lavfi',
            '-i', 'anullsrc',
            '-i', video_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-shortest',
            '-f', 'avi',
            _output_path,
            '-loglevel', 'panic'
        ]

        subprocess.run(_cmd, shell=False, capture_output=False, stdout=None)

        if not os.path.isfile(_output_path):
            logger.warning(f"Add silent sound to video '{video_path}' fail.")
            exit(1)

        return _output_path


