import os

_RESULT_DIR = 'decode/vsr/vi'
_VISUAL_DIR = 'data/processed/visual'
_AUDIO_DIR = 'data/processed/audio'
_VIDEO_DIR = 'data/processed/video'
_ORIGIN_DIR = 'data/processed/origin'
_SUBTITLE_DIR = 'data/processed/subtitle'

_OUTPUT_DIR = os.path.join(_RESULT_DIR, 'output')
_FACE_OUTPUT_DIR = os.path.join(_OUTPUT_DIR, 'face')
_ORIGIN_OUTPUT_DIR = os.path.join(_OUTPUT_DIR, 'origin')

os.makedirs(_SUBTITLE_DIR, exist_ok=True)
os.makedirs(_FACE_OUTPUT_DIR, exist_ok=True)
os.makedirs(_ORIGIN_OUTPUT_DIR, exist_ok=True)

__all__ = [
    '_RESULT_DIR',
    '_VISUAL_DIR',
    '_AUDIO_DIR',
    '_VIDEO_DIR',
    '_ORIGIN_DIR',
    '_SUBTITLE_DIR',
    '_OUTPUT_DIR',
    '_FACE_OUTPUT_DIR',
    '_ORIGIN_OUTPUT_DIR',
]
