import os
import shutil
from typing import List


from src.models.taskers.tasker import Tasker
from src.data.processors.cropper import Cropper
from src.models.utils.logging import get_logger

logger = get_logger(__name__, is_stream=True, log_path=None)


class DemoCropper(Tasker):

    def __init__(self):
        super().__init__()
        self.cropper = Cropper()

    def do(self, samples: List[dict], *args, **kwargs) -> List[dict]:
        _samples = []

        for sample in samples:
            _v_mouth_dir = os.path.join(sample['visual_output_dir'][0], 'mouth')
            _a_mouth_dir = _v_mouth_dir.replace('visual', 'audio')

            os.makedirs(_v_mouth_dir, exist_ok=True)
            os.makedirs(_a_mouth_dir, exist_ok=True)
            _tmp_sample = self.cropper.process(sample, visual_output_dir=_v_mouth_dir)
            if _tmp_sample['id'] is None:
                continue

            _sample = dict()
            _old_audio_path = _tmp_sample['visual_path'][0].replace('visual', 'audio').replace('mp4', 'wav')
            _sample['visual_path'] = os.path.join(_v_mouth_dir, sample['chunk_visual_id'][0] + '.mp4')
            _sample['audio_path'] = _sample['visual_path'].replace('visual', 'audio').replace('mp4', 'wav')
            _sample['visual_num_frames'] = _tmp_sample['visual_num_frames'][0]
            _sample['audio_num_frames'] = _tmp_sample["audio_num_frames"][0]

            shutil.copy(src=_old_audio_path, dst=_sample['audio_path'])
            _idx = _sample['visual_path'].split('@')[-1][:-4]

            _samples.append(_sample)

        if not _samples:
            logger.fatal('No mouth of speakers can be cropped.')
            exit(1)

        return _samples
