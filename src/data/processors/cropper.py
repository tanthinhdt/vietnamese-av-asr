import os
import cv2
import numpy as np
import mediapipe as mp
import moviepy.editor as mpe
from .processor import Processor


class Cropper(Processor):
    """
    This class is used to crop mouth region.
    """
    def __init__(self) -> None:
        self.landmark_detector = mp.solutions.face_mesh.FaceMesh()
        self.mouth_landmark_idxes = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        ]

    def process(
        self, sample: dict,
        visual_output_dir: str,
        padding: int = 96,
    ) -> dict:
        """
        Crop mouth region in video.
        :param sample:              Sample.
        :param visual_output_dir:   Path to directory containing cropped mouth region.
        :param padding:             Padding.
        :return:                    Sample with path to video of cropped mouth region.
        """
        visual_output_path = os.path.join(visual_output_dir, sample["id"][0] + ".mp4")

        if not os.path.exists(visual_output_path):
            mouths = []
            max_width, max_height = 0, 0
            for frame in mpe.VideoFileClip(sample["visual"][0]).iter_frames():
                mouth = self.crop_mouth(frame, padding)
                if mouth is None or mouth.shape[0] == 0 or mouth.shape[1] == 0:
                    continue
                max_width = max(max_width, mouth.shape[1])
                max_height = max(max_height, mouth.shape[0])
                mouths.append(mouth)

            if self.check_output(
                num_cropped=len(mouths),
                sample_fps=sample["fps"][0],
                sample_duration=sample["duration"][0],
            ):
                self.write_video(
                    video_path=visual_output_path,
                    frames=mouths,
                    frame_width=max_width,
                    frame_height=max_height,
                    fps=sample["fps"][0],
                )
            else:
                sample["id"][0] = None

        return sample

    def check_output(
        self, num_cropped: int,
        sample_fps: int,
        sample_duration: int
    ) -> int:
        """
        Check output.
        :param num_cropped:         Number of cropped frames.
        :param sample_fps:          Sample FPS.
        :param sample_duration:     Sample duration.
        :return:                    Whether output is valid.
        """
        if abs(num_cropped / sample_fps - sample_duration) > 0.1:
            return False
        return True

    def crop_mouth(self, frame: np.ndarray, padding: int) -> np.ndarray:
        """
        Crop mouth region in frame.
        :param frame:       Frame.
        :param padding:     Padding.
        :return:            Mouth region.
        """
        # face_landmarks = self.landmark_detector.process(frame).multi_face_landmarks

        # if face_landmarks:
        #     mouth_landmarks = np.array(face_landmarks[0].landmark)[self.mouth_landmark_idxes]
        #     max_x, max_y = 0, 0
        #     min_x, min_y = frame.shape[1], frame.shape[0]
        #     for landmark in mouth_landmarks:
        #         x = int(landmark.x * frame.shape[1])
        #         y = int(landmark.y * frame.shape[0])
        #         max_x, max_y = max(max_x, x), max(max_y, y)
        #         min_x, min_y = min(min_x, x), min(min_y, y)
        #     max_x += padding
        #     max_y += padding
        #     min_x -= padding
        #     min_y -= padding
        #     return frame[min_y:max_y, min_x:max_x]
        
        face_landmarks = self.landmark_detector.process(frame).multi_face_landmarks

        if face_landmarks:
            mouth_landmarks = np.array([
                [landmark.x, landmark.y] for landmark in face_landmarks[0].landmark
            ])[self.mouth_landmark_idxes, :]
            center_x = np.mean(mouth_landmarks[:, 0]) * frame.shape[1]
            min_x = int(center_x - padding / 2)
            max_x = int(center_x + padding / 2)
            center_y = np.mean(mouth_landmarks[:, 1]) * frame.shape[0]
            min_y = int(center_y - padding / 2)
            max_y = int(center_y + padding / 2)
            return frame[min_y:max_y, min_x:max_x]
        return None

    def write_video(
        self, video_path: str,
        frames: list,
        frame_width: int,
        frame_height: int,
        fps: int,
    ) -> None:
        """
        Write video.
        :param video_path:      Path to video.
        :param frames:          Frames.
        :param frame_width:     Frame width.
        :param frame_height:    Frame height.
        :param fps:             FPS.
        """
        mpe.VideoFileClip.write_videofile(
            mpe.ImageSequenceClip(
                [cv2.resize(frame, (frame_width, frame_height)) for frame in frames],
                fps=fps,
            ),
            video_path,
            logger=None,
        )
