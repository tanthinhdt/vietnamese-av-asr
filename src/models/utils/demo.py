import os
import logging

logger = logging.getLogger(__name__)

_DATASET_DIR = 'src/models/dataset/vsr'


def create_demo_mainfest(
        samples_dict: list
):
    """
    Create mainfest file .tsv, prepare to demo
    :return:
    """

    _dir_path = os.path.join(_DATASET_DIR, 'vi')
    os.makedirs(_dir_path, exist_ok=True)
    _tsv_path = os.path.join(_dir_path, 'test.tsv')
    f = open(_tsv_path, mode='w')

    _lines = ['.\n']

    for idx, sample in enumerate(samples_dict):
        _id = "%d/vi-vi" % idx
        _abs_visual_path = sample['visual_path']
        _abs_audio_path = sample['audio_path']

        _line = '\t'.join(map(str, (_id, _abs_visual_path, _abs_audio_path, sample['visual_num_frames'], sample['audio_num_frames'])))
        _lines.append(_line+'\n')

    f.writelines(_lines)

    f.close()
    logger.info("Created mainfest for demo.")
