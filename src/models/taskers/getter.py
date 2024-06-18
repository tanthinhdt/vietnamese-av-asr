import math
import os
import subprocess
import sys

sys.path.append(os.getcwd())

import glob
import json
import time
from typing import Any

from src.models.taskers.tasker import Tasker

_RESULT_DIR = 'decode/vsr/vi'
_VISUAL_DIR = 'data/processed/visual'
_AUDIO_DIR = 'data/processed/audio'
_SUBTITLE_DIR = 'data/processed/subtitle'

_OUTPUT_DIR = os.path.join(_RESULT_DIR, 'outputs')
_MOUTH_DIR = os.path.join(_OUTPUT_DIR, 'mouth')

os.makedirs(_MOUTH_DIR, exist_ok=True)
os.makedirs(_SUBTITLE_DIR, exist_ok=True)

class Reader(Tasker):

    def __init__(self):
        super().__init__()

    def do(self, sample: Any, *args, **kwargs) -> Any:
        decode_file = glob.glob(_RESULT_DIR + '/hypo*.json')[0]
        with open(decode_file, 'r') as f:
            _hypo_dict = json.load(f)
        for utt_id, hypo in zip(_hypo_dict['utt_id'], _hypo_dict['hypo']):
            _id = "%.5d" % int(utt_id.split('/')[0])
            _name = 'visual_' + _id
            _input_path = os.path.join(_VISUAL_DIR, _name + '.mp4')
            _srt_path = os.path.join(_SUBTITLE_DIR, 'subtitle_%s' % _id + '.srt')
            _output_path = os.path.join(_MOUTH_DIR, _name + '.mp4')

            self.create_srt_file(_srt_path, subtitle=hypo)

            _cmd = [
                'ffmpeg', '-y',
                '-i', _input_path,
                '-vf', 'subtitles=%s' % _srt_path,
                _output_path
            ]

            subprocess.run(_cmd, shell=False, capture_output=False, stdout=None)
            exit()


    def create_srt_file(self, srt_path: str, subtitle: str):
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



reader = Reader()
reader.do(None)