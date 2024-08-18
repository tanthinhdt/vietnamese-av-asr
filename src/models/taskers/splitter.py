import os
import subprocess

from src.models.taskers.tasker import Tasker
from src.models.utils.media import get_duration
from src.models.utils.logging import get_logger
from src.models.utils.dirs import *

logger = get_logger(name='Splitter', is_stream=True)


class Splitter(Tasker):
    FPS = 25
    SR = 16_000
    DETACH_VISUAL_CMD = "ffmpeg -y -i %s -an -c:v copy -r 25 -map 0 -f avi %s -loglevel panic"
    DETACH_AUDIO_CMD = "ffmpeg -y -i %s -vn -ac 1 -c:a pcm_s16le -ar 16000 -b:a 192k -map 0 -f wav %s -loglevel panic"
    def __init__(self):
        super().__init__()

    def do(self, metadata_dict: dict, time_interval: int) -> dict:
        _cmd = [
            'ffmpeg', '-y',
            '-loglevel', 'panic',
            '-i', metadata_dict['video_path'],
            '-ss', '%.3f',
            '-t', '%3.f',
            '-c:v', 'libx264',
            '-c:a', 'pcm_s16le',
            '-ac', '1',
            '-r', f'{self.FPS}',
            '-ar', f'{self.SR}',
            '-f', 'avi',
            '%s',
        ]
        os.makedirs(os.path.join(_VIDEO_DIR, 'origin'), exist_ok=True)
        os.makedirs(os.path.join(_VISUAL_DIR, 'origin'), exist_ok=True)
        os.makedirs(os.path.join(_AUDIO_DIR, 'origin'), exist_ok=True)
        _samples = dict()
        _samples['result_video_path'] = metadata_dict['result_video_path']
        i = 0

        if time_interval == -1 or time_interval > int(metadata_dict['duration']):
            time_interval = int(metadata_dict['duration'])
        for timestamp in range(0, int(metadata_dict['duration']), time_interval):
            _video_name = "video_%.5d" % i
            _video_path = os.path.join(
                _VIDEO_DIR, 'origin', _video_name + '.mp4'
            )
            subprocess.run(
                " ".join(_cmd) % (float(timestamp), float(time_interval), _video_path),
                shell=True,
                capture_output=False,
                stdout=None
            )

            if os.path.isfile(_video_path):
                pv_path = _video_path.replace('video', 'visual')
                pa_path = _video_path.replace('video', 'audio').replace('mp4', 'wav')

                _sample = dict()
                _sample['index'] = i
                _sample['chunk_visual_id'] = _video_name.replace('video', 'visual')
                dur = get_duration(_video_path)
                _sample['timestamp'] = (timestamp, timestamp+int(dur))
                _sample['visual_output_dir'] = _VISUAL_DIR
                _sample['visual_path'] = self._detach(
                    cmd=self.DETACH_VISUAL_CMD,
                    video_path=_video_path,
                    output_path=pv_path,
                    fail_msg="Detach visual fail",
                )
                _sample['audio_path'] = self._detach(
                    cmd=self.DETACH_AUDIO_CMD,
                    video_path=_video_path,
                    output_path=pa_path,
                    fail_msg="Detach audio fail",
                )
                _samples[i] = _sample
                i += 1
        if i == 0:
            logger.critical(f"Can not split video '{metadata_dict['video_path']}' into segments.")
            raise RuntimeError()

        return _samples

    def _detach(self, cmd: str,  video_path: str, output_path: str, fail_msg: str):
        command = cmd % (video_path, output_path)
        subprocess.run(command, shell=True, stdout=None)

        if not os.path.isfile(output_path):
            logger.critical(fail_msg)
            raise RuntimeError()

        return output_path