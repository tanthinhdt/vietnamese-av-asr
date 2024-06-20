import os
import subprocess
import json

from src.models.taskers.tasker import Tasker
from src.models.utils.logging import get_logger

logger = get_logger(
    name=__name__,
    is_stream=True,
    log_path=None,
)

class Checker(Tasker):
    """Check condition of sample"""

    _SAMPLE_RATE: int = 16000
    _FRAME_RATE: int = 25
    _EXTENSION: str = '.mp4'

    def __init__(self, duration_threshold: int = 3,  *args, **kwargs):
        """
        Initialize checker

        path_file:
            Path to video file
        args:
            Positional arguments
        kwargs:
            Keyword arguments.
        """
        super().__init__()
        self._SAMPLE_RATE = kwargs.get('sample_rate', self._FRAME_RATE)
        self._FRAME_RATE = kwargs.get('frame_rate', self._SAMPLE_RATE)
        self._EXTENSION = kwargs.get('extension', self._EXTENSION)
        self._threshold_duration: int = duration_threshold

    def do(self, video_path: str, *args, **kwargs) -> dict:

        _stream_dict = self._get_metadata_streams(video_path=video_path)

        metadata_dict = dict()
        metadata_dict['video_path'] = video_path
        metadata_dict['duration'] = _stream_dict['duration']
        metadata_dict['extension'] = os.path.splitext(video_path)[1][1:]
        metadata_dict['has_v'] = 'visual' in _stream_dict
        metadata_dict['has_a'] = 'audio' in _stream_dict
        metadata_dict['v_codec'] = None
        metadata_dict['a_codec'] = None
        if metadata_dict['has_v']:
            metadata_dict['v_codec'] = _stream_dict['visual']['codec_name']
        if metadata_dict['has_a']:
            metadata_dict['a_codec'] = _stream_dict['audio']['codec_name']

        self.post_do(metadata_dict)
        return metadata_dict

    def post_do(self, metadata_dict: dict):
        if metadata_dict['duration'] > self._threshold_duration:
            logger.critical("Duration of video (%.3f) over threshold (%.3f)" % (metadata_dict['duration'], self._threshold_duration))
            exit(1)
        n_con = 0
        if not metadata_dict['has_v']:
            logger.info(f"Video in '{metadata_dict['video_path']}\' has no visual.")
            n_con += 1
        if not metadata_dict['has_a']:
            logger.info(f"Video in '{metadata_dict['video_path']}\' has no audio.")
            n_con += 1
        if n_con == 2:
            logger.critical(f"INVALID. Video in '{metadata_dict['video_path']}\' has no visual and audio.")
            exit(1)

    def _get_metadata_streams(self, video_path: str) -> dict:
        """
        Get metadata of streams

        Return:
             Dictionary contains metadata.
        """
        if not os.path.isfile(path=video_path):
            logger.error(f"The video file '{video_path}' not exist.")
            exit(1)

        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries",
            "stream=codec_type:"
            "stream=codec_name:"
            "stream=index:"
            "stream=sample_rate:"
            "stream=r_frame_rate:"
            "format=duration",
            "-print_format", "json",
            "%s" % video_path
        ]
        stdout = subprocess.run(
                cmd,
                shell=False,
                capture_output=True,
                stdout=None,
                stderr=None,
                cwd=os.getcwd(),
            ).stdout
        result = json.loads(stdout)

        _streams_dict = dict()
        _streams_dict['duration'] = float(result['format']['duration'])

        for stream in result['streams']:
            if stream['codec_type'] == 'video':
                _streams_dict['visual'] = stream
            elif stream['codec_type'] == 'audio':
                _streams_dict['audio'] = stream

        return _streams_dict