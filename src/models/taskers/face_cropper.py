import os
import subprocess
import cv2

from src.models.taskers.tasker import Tasker
from src.models.utils.logging import get_logger
import mediapipe as mdp

logger = get_logger(__name__, is_stream=True, log_path=None)


class FaceCropper(Tasker):
    def __init__(self):
        super().__init__()
        self.mp_face_detection      = mdp.solutions.face_detection
        self.short_range_detector   = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.5, model_selection=0)
        self.full_range_detector    = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.5, model_selection=1)

    def do(self, samples: dict, need_to_crop: bool, *args, **kwargs) -> dict:
        if not need_to_crop:
            return samples
        for sample in samples['samples']:
            os.makedirs(sample['visual_output_dir'][0] + '/face', exist_ok=True)
            cap = cv2.VideoCapture(sample['visual_path'][0])
            face_video_path = sample['visual_path'][0].replace('origin', 'face').replace('mp4', 'avi')
            vOut = cv2.VideoWriter(face_video_path, cv2.VideoWriter_fourcc(*'XVID'), 25, (224, 224))
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                ih, iw = frame.shape[:2]
                imageNumpy = frame
                results = self.full_range_detector.process(imageNumpy)
                if not results.detections:
                    results = self.short_range_detector.process(imageNumpy)
                if results.detections:
                    for detection in results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        bbox = (bboxC.xmin * iw,
                                bboxC.ymin * ih,
                                (bboxC.width + bboxC.xmin) * iw,
                                (bboxC.height + bboxC.ymin) * ih)
                        bbox = [int(_) for _ in bbox]
                        face = imageNumpy[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                        vOut.write(cv2.resize(face, (224, 224)))
                else:
                    pass

            vOut.release()
            cmd = [
                'ffmpeg', '-y',
                '-loglevel', 'panic',
                '-i', face_video_path,
                '-c:v', 'libx264',
                '-f', 'avi',
                face_video_path.replace('avi', 'mp4')
            ]

            subprocess.run(cmd, shell=False, stdout=None, capture_output=False)
            if os.path.isfile(face_video_path.replace('avi', 'mp4')):
                os.remove(path=face_video_path)
                sample['visual_path'] = [face_video_path.replace('avi', 'mp4')]

        return samples