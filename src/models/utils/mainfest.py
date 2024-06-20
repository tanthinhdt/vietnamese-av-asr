import os
import sys
sys.path.append(os.getcwd())

from src.models.utils.dirs import _DATASET_DIR

def create_demo_mainfest(
        samples_dict: list
) -> str:
    """
    Create mainfest file .tsv, prepare to demo
    :return: mainfest file
    """

    _dir_path = os.path.join(_DATASET_DIR, 'vsr', 'vi')
    os.makedirs(_dir_path, exist_ok=True)
    _tsv_path = os.path.join(_dir_path, 'test.tsv')

    _lines = ['.\n']

    for idx, sample in enumerate(sorted(samples_dict, key=lambda x: x['index'])):
        _id = "%d/vi-vi" % idx
        _abs_visual_path = os.path.abspath(sample['visual_path'])
        _abs_audio_path = os.path.abspath(sample['audio_path'])

        _line = '\t'.join(map(str, (_id, _abs_visual_path, _abs_audio_path, sample['visual_num_frames'], sample['audio_num_frames'])))
        _lines.append(_line+'\n')

    f = open(_tsv_path, mode='w')
    f.writelines(_lines)
    f.close()

    return _dir_path
