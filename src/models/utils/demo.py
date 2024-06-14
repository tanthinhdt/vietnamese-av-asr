import glob
import math
import os
import subprocess
import logging

logger = logging.getLogger(__name__)

_PROCESSED_DIR = "data/processed"
_DATASET_DIR = 'src/models/dataset/vsr'


def create_demo_mainfest():
    """
    Create mainfest file .tsv, prepare to demo
    :return:
    """
    _visual_paths = glob.glob(os.path.join(_PROCESSED_DIR, 'visual', '*.mp4'))
    _audio_paths = glob.glob(os.path.join(_PROCESSED_DIR, 'audio', '*.wav'))

    _visual_paths.sort()
    _audio_paths.sort()

    _dir_path = os.path.join(_DATASET_DIR, 'vi')
    os.makedirs(_dir_path, exist_ok=True)
    _tsv_path = os.path.join(_dir_path, 'test.tsv')
    f = open(_tsv_path, mode='w')

    _lines = ['.\n']

    for idx, (visual_path, audio_path) in enumerate(zip(_visual_paths, _audio_paths)):
        _dur_cmd = "ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 %s" % audio_path
        _fr_cmd = "ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of default=noprint_wrappers=1:nokey=1 %s" % visual_path
        _sr_cmd = "ffprobe -v error -select_streams a:0 -of default=noprint_wrappers=1:nokey=1 -show_entries stream=sample_rate %s" % audio_path

        _fr = float(subprocess.run(_fr_cmd, shell=True, capture_output=True).stdout.strip()[:2])
        _sr = float(subprocess.run(_sr_cmd, shell=True, capture_output=True).stdout.strip())
        _dur = float(subprocess.run(_dur_cmd, shell=True, capture_output=True).stdout.strip())

        _nf = math.ceil(_dur * _fr)
        _ns = math.ceil(_dur * _sr)

        _id = "%d/vi-vi" % idx
        _abs_visual_path = os.path.abspath(visual_path)
        _abs_audio_path = os.path.abspath(audio_path)

        _line = '\t'.join(map(str, (_id, _abs_visual_path, _abs_audio_path, _nf, _ns)))
        _lines.append(_line+'\n')

    f.writelines(_lines)

    f.close()
    logger.info("Created mainfest for demo.")
