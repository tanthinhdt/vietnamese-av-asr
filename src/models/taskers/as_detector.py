import os

from typing import List

from src.models.taskers import Tasker
from src.models.utils import get_logger
from src.data.processors.as_extracter import ActiveSpeakerExtracter

logger = get_logger(name=__name__, is_stream=True)


class DemoASDetector(Tasker):

    def __init__(self, time_interval: int = 3):
        super().__init__()
        self.detector = ActiveSpeakerExtracter()
        self.output_dir = 'data/processed'
        self.visual_output_dir = self.output_dir + '/visual/'
        self.audio_output_dir = self.output_dir + '/audio/'
        self.tmp_dir = 'data/interim'
        self.time_interval = time_interval

    def do(self, metadata_dict: dict) -> List[dict]:
        """
        Detect speaker in video.

        metadata_dict:
            Dict contains metadata.
        """
        sample = dict()
        sample['id'] = ['id']
        sample['channel'] = ['face']
        sample['video_id'] = ['video_id']
        sample['video_path'] = [metadata_dict['video_path']]
        sample['demo'] = [True]

        samples = self.detector.process(
            sample,
            output_dir=self.output_dir,
            visual_output_dir=self.visual_output_dir,
            audio_output_dir=self.audio_output_dir,
            tmp_dir=self.tmp_dir,
            log_path=None,
            combine_av=True,
            keep_origin=True,
            time_interval=self.time_interval,
        )

        _samples = []

        for _id, _c_id in zip(samples['id'], samples['chunk_visual_id']):
            if _id is None:
                continue
            _sample = dict()
            _sample['id'] = [_id]
            _sample['visual_path'] = [os.path.join(self.visual_output_dir, samples['channel'][0], _c_id) + '.mp4']
            _sample['visual_output_dir'] = [self.visual_output_dir]
            _sample['chunk_visual_id'] = [_c_id]
            _sample['visual_fps'] = [self.detector.V_FPS]
            _sample['visual_num_frames'] = [self.detector.V_FPS * self.detector.DURATION]
            _sample['audio_num_frames'] = [self.detector.A_FPS * self.detector.DURATION]
            _samples.append(_sample)

        assert _samples, logger.warning('No speaker is detected in video.')

        return _samples

