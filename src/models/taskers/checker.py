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

    def __init__(self, *args, **kwargs):
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

    def do(self, video_path: str, *args, **kwargs) -> dict:

        _stream_dict = self._get_metadata_streams(video_path=video_path)

        metadata_dict = dict()
        metadata_dict['video_path'] = video_path
        metadata_dict['extension'] = os.path.splitext(video_path)[1][1:]
        metadata_dict['match_ext'] = self._check_ext(video_path=video_path)
        metadata_dict['has_v'] = self._check_visual(_stream_dict)
        metadata_dict['has_a'] = self._check_audio(_stream_dict)
        metadata_dict['match_fr'] = self._check_frame_rate(_stream_dict)
        metadata_dict['match_sr'] = self._check_sample_rate(_stream_dict)

        self.post_do(metadata_dict)

        return metadata_dict

    def post_do(self, metadata_dict: dict):
        if not metadata_dict['has_v']:
            logger.fatal(f"Video in '{metadata_dict['video_path']}\' has no visual.")
            exit(1)
        if not metadata_dict['has_a']:
            logger.fatal(f"Video in '{metadata_dict['video_path']}\' has no audio.")
            exit(1)

    def _check_is_file(self, video_path: str) -> bool:
        """
        Check path file whether exist.

        Return:
            True if exist and otherwise
        """
        return os.path.isfile(path=video_path)

    def _check_ext(self, video_path: str):
        """
        Check extension of file

        ext:
            Extension the file should be match
        Return:
            True if match and otherwise
        """
        return os.path.splitext(video_path)[1] == self._EXTENSION

    def _get_metadata_streams(self, video_path: str) -> dict:
        """
        Get metadata of streams

        Return:
             Dictionary contains metadata.
        """
        if not self._check_is_file(video_path=video_path):
            logger.error(f"The video file '{video_path}' not exist.")
            exit(1)
        if not self._check_ext(video_path=video_path):
            logger.warning(f"Highly recommend use video file with extension '{self._EXTENSION}'.")
            exit(1)
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries",
            "stream=codec_type:"
            "stream=codec_name:"
            "stream=index:"
            "stream=sample_rate:"
            "stream=r_frame_rate",
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

        for stream in result['streams']:
            if stream['codec_type'] == 'video':
                _streams_dict['visual'] = stream
            elif stream['codec_type'] == 'audio':
                _streams_dict['audio'] = stream

        return _streams_dict

    def _check_visual(self, metadata_streams: dict):
        """
        Check whether stream has visual.

        metadata_streams:
             Metadata of streams.
        Return:
             True if it has visual, otherwise.
        """
        return 'visual' in metadata_streams

    def _check_audio(self, metadata_streams: dict):
        """
       Check whether stream has audio.

       metadata_streams:
            Metadata of streams.
       Return:
            True if it has audio, otherwise.
       """
        return 'audio' in metadata_streams

    def _check_frame_rate(self, metadata_streams: dict):
        """
        Check origin frame rate whether match expected one.

        metadata_streams:
            Metadata of streams.
        Return:
            True if matched, otherwise
        """
        if self._check_visual(metadata_streams):
            return int(metadata_streams['visual']['r_frame_rate'][:2]) == self._FRAME_RATE
        return False

    def _check_sample_rate(self,metadata_streams):
        """
       Check origin sample rate whether match expected one.

       metadata_streams:
           Metadata of streams.
       Return:
           True if matched, otherwise
       """
        if self._check_audio(metadata_streams):
            return int(metadata_streams['audio']['sample_rate']) == self._SAMPLE_RATE
        return False
