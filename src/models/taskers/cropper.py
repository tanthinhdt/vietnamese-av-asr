import os
import shutil
from typing import List

from src.models.taskers.tasker import Tasker
from src.data.processors.cropper import Cropper


class DemoCropper(Tasker):

    def __init__(self):
        super().__init__()
        self.cropper = Cropper()

    def do(self, samples: List[dict], *args, **kwargs) -> List[dict]:
        _samples = []

        for sample in samples:
            _tmp_sample = self.cropper.process(sample, visual_output_dir=sample['visual_output_dir'][0])
            if _tmp_sample['id'] is None:
                continue

            _sample = dict()
            _old_audio_path = _tmp_sample['visual_path'][0].replace('visual', 'audio').replace('mp4', 'wav')
            _sample['visual_path'] = os.path.join(sample['visual_output_dir'][0], sample['chunk_visual_id'][0] + '.mp4')
            _sample['audio_path'] = _sample['visual_path'].replace('visual', 'audio').replace('mp4', 'wav')
            _sample['visual_num_frames'] = _tmp_sample['visual_num_frames'][0]
            _sample['audio_num_frames'] = _tmp_sample["audio_num_frames"][0]

            shutil.copy(src=_old_audio_path, dst=_sample['audio_path'])
            _idx = _sample['visual_path'].split('@')[-1][:-4]

            _new_visual_path = os.path.join(sample['visual_output_dir'][0],f'visual_{_idx}.mp4')
            _new_audio_path = _new_visual_path.replace('visual', 'audio').replace('mp4', 'wav')
            os.rename(
                src=_sample['visual_path'],
                dst=_new_visual_path
            )
            os.rename(
                src=_sample['audio_path'],
                dst=_new_audio_path
            )
            _sample['visual_path'] = _new_visual_path
            _sample['audio_path'] = _new_audio_path

            _samples.append(_sample)

        return _samples
