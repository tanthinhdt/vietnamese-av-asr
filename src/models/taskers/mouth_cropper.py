import os
import subprocess
import math
import cv2
import torchvision
import numbers

from src.models.utils.logging import get_logger
from src.models.taskers.tasker import Tasker
from src.models.utils.media import get_duration, get_sr, get_fps
from ..detector.detector import LandmarksDetector
from ..detector.video_process import VideoProcess

logger = get_logger(__name__, is_stream=True, log_path=None)


class MouthCropper(Tasker):
    def __init__(self, convert_gray=False):
        self.landmarks_detector = LandmarksDetector()
        self.video_process = VideoProcess(convert_gray=convert_gray)

    def load_data(self, data_filename):
        video = self.load_video(data_filename)
        landmarks = self.landmarks_detector(video)
        video = self.video_process(video, landmarks)
        if video is None:
            return
        return video

    def load_video(self, data_filename):
        return torchvision.io.read_video(data_filename, pts_unit="sec")[0].numpy()

    def write_video(self, video_path, mouth_video_path):
        sequence = self.load_data(data_filename=video_path)
        if sequence is None:
            return
        vOut = cv2.VideoWriter(mouth_video_path, cv2.VideoWriter_fourcc(*'XVID'), 25, (88, 88))
        for image in sequence:
            vOut.write(image)
        vOut.release()
        cmd = [
            'ffmpeg', '-y',
            '-loglevel', 'panic',
            '-i', mouth_video_path,
            '-c:v', 'libx264',
            '-f', 'avi',
            new_mouth_path := mouth_video_path.replace('avi', 'mp4')
        ]

        subprocess.run(cmd, shell=False, stdout=None, capture_output=False)
        if os.path.isfile(new_mouth_path):
            os.remove(path=mouth_video_path)

        return new_mouth_path

    def do(self, samples: dict, need_to_crop: bool = True, *args, **kwargs) -> dict:
        for k in filter(lambda x: isinstance(x, numbers.Number), samples):
            if need_to_crop:
                chunk_visual_id = samples[k]['chunk_visual_id']
                _v_mouth_dir = os.path.join(samples[k]['visual_output_dir'], 'mouth')
                os.makedirs(_v_mouth_dir, exist_ok=True)
                _v_mouth_path = os.path.join(_v_mouth_dir, chunk_visual_id + '.avi')
                samples[k]['visual_path'] = self.write_video(video_path=samples[k]['visual_path'], mouth_video_path=_v_mouth_path)
                if samples[k]['visual_path'] is None:
                    samples.pop(k)
                    continue
            else:
                samples[k]['visual_path'] = samples[k]['visual_path']

            dur = get_duration(samples[k]['visual_path'])
            fps = get_fps(samples[k]['visual_path'])
            sr = get_sr(samples[k]['audio_path'])
            samples[k]['visual_num_frames'] = math.ceil(dur * fps)
            samples[k]['audio_num_frames'] = math.ceil(dur * sr)
            samples[k].pop('visual_output_dir')
            samples[k].pop('audio_output_dir')

        if not samples:
            logger.critical('No mouth of speakers can be cropped.')
            exit(1)

        return samples