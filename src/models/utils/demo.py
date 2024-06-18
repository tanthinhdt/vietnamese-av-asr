import os
import sys
import logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(lineno)d in <%(funcName)s> | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

_DATASET_DIR = 'src/models/dataset'


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

    for idx, sample in enumerate(samples_dict):
        _id = "%d/vi-vi" % idx
        _abs_visual_path = os.path.abspath(sample['visual_path'])
        _abs_audio_path = os.path.abspath(sample['audio_path'])

        _line = '\t'.join(map(str, (_id, _abs_visual_path, _abs_audio_path, sample['visual_num_frames'], sample['audio_num_frames'])))
        _lines.append(_line+'\n')

    f = open(_tsv_path, mode='w')
    f.writelines(_lines)
    f.close()

    logger.info("Created mainfest for demo.")

    return _dir_path
