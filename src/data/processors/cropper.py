import os
import subprocess
import cv2
import numpy as np
import mediapipe as mp
import moviepy.editor as mpe

from .processor import Processor
from src.data.utils.logger import get_logger


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
        log_path: str = None,
    ) -> dict:
        """
        Crop mouth region of speaker in video.

        sample: 
            Dict contains metadata.
        visual_output_dir:   
            Path to directory containing cropped mouth region.
        padding:  
            Padding.
        return:      
            Metadata of processed sample.
        """
        print()
        logger = get_logger(
            name=__name__,
            log_path=log_path,
            is_stream=False,
        )
        logger_ = get_logger(
            log_path=log_path,
            is_stream=False,
            format='%(message)s'
        )

        logger_.info('-'*35 + f"Cropper processing visual id '{sample['chunk_visual_id'][0]}'" + '-'*35)
        _visual_output_path     = os.path.join(visual_output_dir, sample["chunk_visual_id"][0] + "tmp.mp4")
        visual_output_path      = os.path.join(visual_output_dir, sample["chunk_visual_id"][0] + ".mp4")

        if not os.path.exists(_visual_output_path):
            mouths = []
            max_width, max_height = 0, 0

            logger.info('Crop mouth')
            for frame in mpe.VideoFileClip(sample["visual_path"][0]).iter_frames(fps=25):
                mouth = self.crop_mouth(frame, padding)
                if mouth is None or mouth.shape[0] == 0 or mouth.shape[1] == 0:
                    continue
                max_width = max(max_width, mouth.shape[1])
                max_height = max(max_height, mouth.shape[0])
                mouths.append(mouth)
                
            logger.info('Check output')
            if self.check_output(
                num_cropped=len(mouths),
                sample_fps=sample["visual_fps"][0],
                sample_duration=sample["visual_num_frames"][0] / sample["visual_fps"][0],
            ):                
                logger.info('Write cropped')
                self.write_video(
                    video_path=_visual_output_path,
                    frames=mouths,
                    frame_width=max_width,
                    frame_height=max_height,
                    fps=sample["visual_fps"][0],
                )
                logger.info('Normalize 3s')
                command = "ffmpeg -y -i %s -an -c:v libx264 -ss %s -t %s -map 0 -f mp4 -loglevel panic %s" % \
                            (_visual_output_path, "00:00:00.00000", "00:00:03.00000", visual_output_path)
                subprocess.run(command, shell=True, stdout=None)
                os.remove(path=_visual_output_path)
            else:
                logger.info('No cropped')
                sample["id"][0] = None

        logger_.info('*'*50 + 'Cropper done.' + '*'*50)
        return sample

    def check_output(
        self, num_cropped: int,
        sample_fps: int,
        sample_duration: int
    ) -> int:
        """
        Check output.

        num_cropped:     
            Number of cropped frames.
        sample_fps:
            Sample FPS.
        sample_duration:    
            Sample duration.
        return:
            Whether output is valid.
        """
        if abs(num_cropped / sample_fps - sample_duration) > 0.1:
            return False
        return True

    def crop_mouth(self, frame: np.ndarray, padding: int) -> np.ndarray:
        """
        Crop mouth region in frame.

        frame: 
            Frame.
        padding:  
            Padding.
        return:
            Mouth region.
        """        
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
        video_path: 
            Path to video.
        frames: 
            Frames.
        frame_width:
            Frame width.
        frame_height:
            Frame height.
        fps:
            FPS.
        """
        mpe.ImageSequenceClip(
            sequence=[cv2.resize(frame, (frame_width, frame_height)) for frame in frames],
            fps=fps
        ).write_videofile(video_path,fps)