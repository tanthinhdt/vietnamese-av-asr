import os
import shutil

_DATASET_DIR = 'src/models/dataset'
_DECODE_DIR = 'decode/vsr/vi'
_VISUAL_DIR = 'data/processed/visual'
_AUDIO_DIR = 'data/processed/audio'
_VIDEO_DIR = 'data/processed/video'
_ORIGIN_DIR = 'data/processed/origin'
_SUBTITLE_DIR = 'data/processed/subtitle'
_FINAL_RESULT_DIR = 'results'


_OUTPUT_DIR = os.path.join(_DECODE_DIR, 'output')
_FACE_OUTPUT_DIR = os.path.join(_OUTPUT_DIR, 'face')
_ORIGIN_OUTPUT_DIR = os.path.join(_OUTPUT_DIR, 'origin')

os.makedirs(_SUBTITLE_DIR, exist_ok=True)
os.makedirs(_FACE_OUTPUT_DIR, exist_ok=True)
os.makedirs(_ORIGIN_OUTPUT_DIR, exist_ok=True)
os.makedirs(_FINAL_RESULT_DIR, exist_ok=True)

__all__ = [
    '_DECODE_DIR',
    '_VISUAL_DIR',
    '_AUDIO_DIR',
    '_VIDEO_DIR',
    '_ORIGIN_DIR',
    '_SUBTITLE_DIR',
    '_OUTPUT_DIR',
    '_FACE_OUTPUT_DIR',
    '_ORIGIN_OUTPUT_DIR',
    '_FINAL_RESULT_DIR',
    '_DATASET_DIR',
    'clean_dirs',
]


def clean_dirs(*dirs):
    for _dir in dirs:
        if os.path.isdir(_dir):
            shutil.rmtree(_dir, ignore_errors=True)
