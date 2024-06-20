import os
import shutil

_DATASET_DIR = 'src/models/dataset'
_DECODE_DIR = 'decode/vsr/vi'
_VISUAL_DIR = 'data/processed/visual'
_AUDIO_DIR = 'data/processed/audio'
_VIDEO_DIR = 'data/processed/video'
_ORIGIN_DIR = 'data/processed/origin'
_FINAL_RESULT_DIR = 'results'

os.makedirs(_FINAL_RESULT_DIR, exist_ok=True)

__all__ = [
    '_DECODE_DIR',
    '_VISUAL_DIR',
    '_AUDIO_DIR',
    '_VIDEO_DIR',
    '_ORIGIN_DIR',
    '_FINAL_RESULT_DIR',
    '_DATASET_DIR',
    'clean_dirs',
]


def clean_dirs(*dirs):
    for _dir in dirs:
        if os.path.isdir(_dir):
            shutil.rmtree(_dir, ignore_errors=True)