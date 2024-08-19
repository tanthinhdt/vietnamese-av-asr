import os
import subprocess
import sys

sys.path.append(os.getcwd())

import glob
import json
import time
from typing import Any

from src.models.taskers.tasker import Tasker
from src.models.utils.dirs import *
from src.models.utils.logging import get_logger

logger = get_logger(name='Embedder', is_stream=True,)


class Embedder(Tasker):

    def __init__(self):
        super().__init__()

    def do(self, samples: dict, *args, **kwargs) -> Any:
        _f_output_paths = []
        _o_output_paths = []
        decode_file = glob.glob(_DECODE_DIR + '/hypo.json')[0]

        with open(decode_file, 'r') as f:
            _hypo_dict = json.load(f)

        _output_path = os.path.join(_FINAL_RESULT_DIR, 'output.mp4')
        _srt_path = os.path.join(_FINAL_RESULT_DIR, 'subtitle.srt')
        if not any(_hypo_dict['hypo']):
            logger.critical('Empty transcript is predicted')
            raise RuntimeError()
        f = open(_srt_path, 'w')
        for utt_id, hypo in sorted(zip(_hypo_dict['utt_id'], _hypo_dict['hypo']), key=lambda x: x[0]):
            index = int(utt_id.split('/')[0])
            timestamp = samples[index]['timestamp']
            self._append_to_srt(f=f, index=index, subtitle=hypo, timestamp=timestamp)
        f.close()
        self._embed_subtitle(samples['result_video_path'], _srt_path, _output_path)

        return _output_path

    def _embed_subtitle(
            self,
            video_path: str,
            subtitle_path: str,
            output_path: str,
    ) -> str:
        if not os.path.isfile(video_path):
            logger.critical("Video file is not exist.")
            raise RuntimeError()
        if not os.path.isfile(subtitle_path):
            logger.critical("Subtitle file is not exist.")
            raise RuntimeError()

        _cmd = [
            'ffmpeg', '-y',
            '-loglevel', 'panic',
            '-i', video_path,
            '-vf', f"subtitles={subtitle_path}:force_style='PrimaryColour=&HFFFFFF,BorderStyle=4,BackColour=0'",
            output_path,
        ]

        subprocess.run(_cmd, shell=False, capture_output=False, stdout=None)

        if not os.path.isfile(output_path):
            logger.warning(f"Add subtitle into video '{video_path}' fail.")
            raise RuntimeError()

        return output_path

    def _append_to_srt(self, f, index: int, subtitle: str, timestamp: tuple):
        _start_time_stamp = time.strftime('%H:%M:%S,000', time.gmtime(timestamp[0]))
        _end_time_stamp = time.strftime('%H:%M:%S,000', time.gmtime(timestamp[1]))
        f.write(str(index) + '\n')
        f.write(_start_time_stamp + ' --> ' + _end_time_stamp + '\n')
        f.write(subtitle + '\n')
        f.write('\n')