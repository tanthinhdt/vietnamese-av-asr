import numbers
import os

from src.models.utils.dirs import _DATASET_DIR


def create_demo_manifest(
        samples_dict: dict
) -> str:
    """
    Create manifest file .tsv, prepare to demo
    :return: manifest file
    """

    _dir_path = os.path.join(_DATASET_DIR, 'vsr', 'vi')
    os.makedirs(_dir_path, exist_ok=True)
    _tsv_path = os.path.join(_dir_path, 'test.tsv')

    _lines = ['.\n']

    for i in samples_dict.keys():
        if not isinstance(i, numbers.Number):
            continue
        _id = "%d/vi-vi" % samples_dict[i]['index']
        _abs_visual_path = os.path.abspath(samples_dict[i]['visual_path'])
        _abs_audio_path = os.path.abspath(samples_dict[i]['audio_path'])

        _line = '\t'.join(map(str, (_id, _abs_visual_path, _abs_audio_path, samples_dict[i]['visual_num_frames'], samples_dict[i]['audio_num_frames'])))
        _lines.append(_line+'\n')

    f = open(_tsv_path, mode='w')
    f.writelines(_lines)
    f.close()

    return _dir_path