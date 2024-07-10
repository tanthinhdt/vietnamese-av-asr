import math
import os
import subprocess

from src.models.taskers.tasker import Tasker
from src.models.utils.media import get_duration
from src.models.utils.logging import get_logger
from src.models.utils.dirs import *

logger = get_logger(name=__name__, log_path=None, is_stream=True)


class Splitter(Tasker):
    FPS = 25
    SR = 16_000

    def __init__(self):
        super().__init__()

    def do(self, metadata_dict: dict, time_interval: int) -> dict:
        _cmd = [
            'ffmpeg', '-y',
            '-loglevel', 'panic',
            '-i', metadata_dict['video_path'],
            '-ss', '%.3f',
            '-t', '%.3f',
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
        _samples['samples'] = []
        i = 0

        for i, timestamp in enumerate(range(0, int(metadata_dict['duration']), time_interval)):
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
                _sample['id'] = [i]
                _sample['chunk_visual_id'] = [_video_name.replace('video', 'visual')]
                dur = get_duration(_video_path)
                _sample['timestamp'] = [(timestamp, timestamp+int(dur))]
                _sample['visual_output_dir'] = [_VISUAL_DIR]
                _sample['audio_output_dir'] = [_AUDIO_DIR]
                _sample['visual_fps'] = [self.FPS]
                _sample['visual_num_frames'] = [math.ceil(dur * self.FPS)]
                _sample['audio_num_frames'] = [math.ceil(dur * self.SR)]
                if metadata_dict['has_v']:
                    _sample['visual_path'] = [self._detach_visual(
                        video_path=_video_path,
                        output_path=pv_path,
                    )]
                else:
                    _sample['visual_path'] = [self._create_placeholder_visual(file=pv_path, duration=dur)]

                if metadata_dict['has_a']:
                    _sample['audio_path'] = [self._detach_audio(
                        video_path=_video_path,
                        output_path=pa_path
                    )]
                else:
                    _sample['audio_path'] = [self._create_placeholder_audio(file=pa_path, duration=dur)]

                _samples['samples'].append(_sample)

                i += 1

        return _samples

    def _create_placeholder_audio(self, file: str, duration: float):
        _cmd = [
            'ffmpeg', '-y',
            '-loglevel', 'panic',
            '-f', 'lavfi',
            '-i', 'anullsrc=channel_layout=mono:sample_rate=%d' % self.SR,
            '-ss', '0.000',
            '-to', '%.3f' % duration,
            '-f', 'wav',
            file
        ]

        subprocess.run(_cmd, shell=False, capture_output=False, stdout=None)

        if not os.path.isfile(file):
            logger.critical(f"Create placeholder audio for non-audio media file fail.")
            exit(1)

        return file

    def _create_placeholder_visual(self, file: str, duration: float):
        _cmd = [
            'ffmpeg', '-y',
            '-loglevel', 'panic',
            '-f', 'lavfi',
            '-i', 'color=c=black',
            '-c:v', 'libx264',
            '-t', '%.3f' % duration,
            '-r', '%d' % self.FPS,
            '-pix_fmt', 'yuv420p',
            '-vf', 'scale=96:96',
            '-f', 'avi',
            file
        ]

        subprocess.run(_cmd, shell=False, capture_output=False, stdout=None)

        if not os.path.isfile(file):
            logger.critical(f"Create placeholder visual for non-visual media file fail.")
            exit(1)

        return file

    def _detach_visual(self, video_path: str, output_path: str,):
        command = "ffmpeg -y -i %s -an -c:v copy -r 25 -map 0 -f avi %s -loglevel panic" % \
                  (video_path, output_path)
        subprocess.run(command, shell=True, stdout=None)

        if not os.path.isfile(output_path):
            logger.critical(f"Detach visual fail")
            exit(1)

        return output_path

    def _detach_audio(self, video_path: str, output_path: str,):
        command = "ffmpeg -y -i %s -vn -ac 1 -c:a pcm_s16le -ar 16000 -b:a 192k -map 0 -f wav %s -loglevel panic" % \
                  (video_path, output_path)
        subprocess.run(command, shell=True, stdout=None)

        if not os.path.isfile(output_path):
            logger.critical(f"Detach audio fail")
            exit(1)

        return output_path