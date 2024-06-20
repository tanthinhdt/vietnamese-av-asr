import math
import os
import shutil
from typing import List


from src.models.taskers.tasker import Tasker
from src.data.processors.cropper import Cropper
from src.models.utils.logging import get_logger
from src.models.utils.media import get_duration, get_sr, get_fps

logger = get_logger(__name__, is_stream=True, log_path=None)


class DemoCropper(Tasker):

    def __init__(self):
        super().__init__()
        self.cropper = Cropper()

    def do(self, samples: dict, need_to_crop: bool = True, *args, **kwargs) -> dict:
        _samples = dict()
        _samples['result_video_path'] = samples['result_video_path']

        i = 0
        for sample in samples['samples']:
            _v_mouth_dir = os.path.join(sample['visual_output_dir'][0], 'mouth')
            _a_mouth_dir = os.path.join(sample['audio_output_dir'][0], 'mouth')
            sample['chunk_visual_id'][0] = sample['visual_path'][0].split('/')[-1][:-4]

            os.makedirs(_v_mouth_dir, exist_ok=True)
            os.makedirs(_a_mouth_dir, exist_ok=True)
            if need_to_crop:
                _tmp_sample = self.cropper.process(sample, visual_output_dir=_v_mouth_dir)
                if _tmp_sample['id'] is None:
                    continue
            else:
                _tmp_sample = dict()
                _tmp_sample['visual_path'] = [os.path.join(_v_mouth_dir, sample['chunk_visual_id'][0]) + '.mp4']
                shutil.copy(src=sample['visual_path'][0], dst=_tmp_sample['visual_path'][0])
                _tmp_sample['visual_path'] = sample['visual_path']

            _sample = dict()
            _sample['index'] = i
            _sample['timestamp'] = sample['timestamp'][0]
            _old_audio_path = _tmp_sample['visual_path'][0].replace('visual', 'audio').replace('mp4', 'wav')
            _sample['visual_path'] = os.path.join(_v_mouth_dir, sample['chunk_visual_id'][0] + '.mp4')
            _sample['audio_path'] = _sample['visual_path'].replace('visual', 'audio').replace('mp4', 'wav')
            dur = get_duration(_sample['visual_path'])
            fps = get_fps(_sample['visual_path'])
            sr = get_sr(_old_audio_path)
            _sample['visual_num_frames'] = math.ceil(dur * fps)
            _sample['audio_num_frames'] = math.ceil(dur * sr)

            shutil.copy(src=_old_audio_path, dst=_sample['audio_path'])

            _samples[i] = _sample
            i += 1

        if not _samples:
            logger.critical('No mouth of speakers can be cropped.')
            exit(1)

        return _samples