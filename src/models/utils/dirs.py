import os
import shutil

_DATASET_DIR = 'src/models/dataset'
_DECODE_DIR = 'decode/vsr/vi'
_VISUAL_DIR = 'data/processed/visual'
_AUDIO_DIR = 'data/processed/audio'
_VIDEO_DIR = 'data/processed/video'
_EXTERNAL_DIR = 'data/external'
_FINAL_RESULT_DIR = 'results'
_OUTPUT_DIR = 'outputs'
_FLAGGED_DIR = 'flagged'

os.makedirs(_DATASET_DIR, exist_ok=True)
os.makedirs(_DECODE_DIR, exist_ok=True)
os.makedirs(_FINAL_RESULT_DIR, exist_ok=True)
os.makedirs(_EXTERNAL_DIR, exist_ok=True)
os.makedirs(_VIDEO_DIR, exist_ok=True)
os.makedirs(_VISUAL_DIR, exist_ok=True)
os.makedirs(_AUDIO_DIR, exist_ok=True)

__all__ = [
    '_DECODE_DIR',
    '_VISUAL_DIR',
    '_AUDIO_DIR',
    '_VIDEO_DIR',
    '_EXTERNAL_DIR',
    '_FINAL_RESULT_DIR',
    '_DATASET_DIR',
    '_OUTPUT_DIR',
    '_FLAGGED_DIR',
]


def clean_dirs():
    dirs = [
        _DATASET_DIR,
        _AUDIO_DIR,
        _VISUAL_DIR,
        _VIDEO_DIR,
        _OUTPUT_DIR,
        _FLAGGED_DIR,
    ]
    for _dir in dirs:
        if os.path.isdir(_dir):
            shutil.rmtree(_dir, ignore_errors=True)