import os
import subprocess
import sys

sys.path.append(os.getcwd())

import glob
import json
import time
from typing import Any, List

from src.models.taskers.tasker import Tasker
from src.models.utils.dirs import *
from src.models.utils.logging import get_logger

logger = get_logger(
    name=__name__,
    log_path=None,
    is_stream=True,
)


class Combiner(Tasker):

    def __init__(self):
        super().__init__()

    def do(self, sample: Any = None, *args, **kwargs) -> Any:
        _f_output_paths = []
        _m_output_paths = []
        _o_output_paths = []
        decode_file = glob.glob(_DECODE_DIR + '/hypo*.json')[0]

        with open(decode_file, 'r') as f:
            _hypo_dict = json.load(f)

        for utt_id, hypo in sorted(zip(_hypo_dict['utt_id'], _hypo_dict['hypo']), key=lambda x: x[0]):
            _id = "%.5d" % int(utt_id.split('/')[0])
            _name = 'chunk@origin@face@video_id@' + _id
            _srt_path = os.path.join(_SUBTITLE_DIR, 'subtitle_%s' % _id + '.srt')
            self._prepare_srt_file(_srt_path, subtitle=hypo)

            _video_path = os.path.join(_ORIGIN_DIR, 'face', _name + '.mp4')
            _output_path = os.path.join(_ORIGIN_OUTPUT_DIR, _name + '.mp4')
            _o_output_paths.append(self._add_subtitle(_video_path, _srt_path, _output_path))

            _video_path = _video_path.replace('origin', 'video')
            _output_path = _output_path.replace('origin', 'face')
            _f_output_paths.append(self._add_subtitle(_video_path, _srt_path, _output_path))

        _o_concat = os.path.join(_ORIGIN_OUTPUT_DIR, 'origin_concat.txt')
        _f_concat = os.path.join(_FACE_OUTPUT_DIR, 'face_concat.txt')
        self._prepare_concat_file(_o_concat, _o_output_paths)
        self._prepare_concat_file(_f_concat, _f_output_paths)
        _o_final_output = os.path.join(_FINAL_RESULT_DIR, 'final_origin_output.mp4')
        _f_final_output = os.path.join(_FINAL_RESULT_DIR, 'final_face_output.mp4')

        self._concat_videos(concat_file=_o_concat, output_path=_o_final_output)
        self._concat_videos(concat_file=_f_concat, output_path=_f_final_output)

        return (
            os.path.abspath(_o_final_output),
            os.path.abspath(_f_final_output)
        )

    def post_do(self, clear_framents: bool = True):
        dirs = [
            _DATASET_DIR,
            _AUDIO_DIR,
            _VISUAL_DIR,
            _VIDEO_DIR,
            _ORIGIN_DIR,
            _SUBTITLE_DIR,
            _OUTPUT_DIR,
        ]

        if clear_framents:
            clean_dirs(*dirs)

    def _add_subtitle(
            self,
            video_path: str,
            subtitle_path: str,
            output_path: str,
    ) -> str:
        if not os.path.isfile(video_path):
            logger.critical("Video file is not exist.")
            exit(1)
        if not os.path.isfile(subtitle_path):
            logger.critical("Subtitle file is not exist.")
            exit(1)

        _cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-vf', 'subtitles=%s' % subtitle_path,
            output_path,
            '-loglevel', 'panic'
        ]

        subprocess.run(_cmd, shell=False, capture_output=False, stdout=None)

        if not os.path.isfile(output_path):
            logger.warning(f"Add subtitle into video '{video_path}' fail.")
            exit(1)

        return output_path

    def _prepare_srt_file(self, srt_path: str, subtitle: str):
        _subtitle_split = subtitle.split(' ')
        _len_sub = len(_subtitle_split)
        _time_interval = 1
        _duration = 3
        f = open(srt_path, 'w')

        for idx, i in enumerate(range(0, _duration, _time_interval)):
            _start_time_stamp = time.strftime('%H:%M:%S,000', time.gmtime(i))
            _end_time_stamp = time.strftime('%H:%M:%S,000', time.gmtime(i+_time_interval))
            _sub_trans = ' '.join(_subtitle_split[i: _len_sub//3*(i+_time_interval)])
            f.write(str(idx) + '\n')
            f.write(_start_time_stamp + ' --> ' + _end_time_stamp + '\n')
            f.write(_sub_trans + '\n')
            f.write('\n')
        f.close()

    def _prepare_concat_file(self, concat_file: str, video_files: List[str]):
        f = open(concat_file, 'w')
        video_files.sort()
        for video_file in video_files:
            _short_path = os.path.abspath(video_file)
            _line = f"file {_short_path}"
            f.write(_line)
            f.write('\n')
        f.close()

    def _concat_videos(self, concat_file: str, output_path: str):
        if not os.path.isfile(concat_file):
            logger.critical(f"'{concat_file}' is not exist.")

        _cmd = [
            'ffmpeg', '-y',
            '-safe', '0',
            '-f', 'concat',
            '-i', concat_file,
            '-c', 'copy',
            output_path,
            '-loglevel', 'panic',
        ]

        subprocess.run(_cmd, shell=False, capture_output=False)

        if not os.path.isfile(output_path):
            logger.critical(f"Concat videos into video '{output_path}' fail.")
