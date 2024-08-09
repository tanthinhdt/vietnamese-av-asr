import os
import subprocess
import math
import cv2
import torchvision
import numbers
import numpy as np

import mediapipe as mp
from src.models.utils.media import get_duration, get_sr, get_fps
from src.models.utils.logging import get_logger

logger = get_logger('Cropper', is_stream=True, log_path=None)


class MouthCropper:
    def __init__(
            self,
            mean_face_path="mean_face.npy",
            crop_width=88,
            crop_height=88,
            start_idx=3,
            stop_idx=4,
            window_margin=12,
            convert_gray=False,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.mp_face_detection = mp.solutions.face_detection
        self.short_range_detector = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5,
                                                                         model_selection=0)
        self.full_range_detector = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=1)

        self.reference = np.load(
            os.path.join(os.path.dirname(__file__), mean_face_path)
        )
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.start_idx = start_idx
        self.stop_idx = stop_idx
        self.window_margin = window_margin
        self.convert_gray = convert_gray

    def load_data(self, data_filename: str):
        video_frames = self.load_video(data_filename)
        landmarks = self.landmarks_detector(video_frames)
        video = self.video_process(video_frames, landmarks)
        if video is None:
            return
        return video

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

    def do(self, samples: dict, need_to_crop: bool = True) -> dict:
        for k in filter(lambda x: isinstance(x, numbers.Number), samples):
            if need_to_crop:
                _v_mouth_dir = os.path.join(samples[k]['visual_output_dir'], 'mouth')
                os.makedirs(_v_mouth_dir, exist_ok=True)
                _v_mouth_path = os.path.join(_v_mouth_dir, samples[k]['chunk_visual_id'] + '.avi')
                samples[k]['visual_path'] = self.write_video(
                    video_path=samples[k]['visual_path'],
                    mouth_video_path=_v_mouth_path
                )
                if samples[k]['visual_path'] is None:
                    samples.pop(k)
                    continue

            dur = get_duration(samples[k]['visual_path'])
            fps = get_fps(samples[k]['visual_path'])
            sr = get_sr(samples[k]['audio_path'])
            samples[k]['visual_num_frames'] = math.ceil(dur * fps)
            samples[k]['audio_num_frames'] = math.ceil(dur * sr)
            samples[k].pop('visual_output_dir')

        if len(samples) == 1:
            logger.critical('No mouth of speakers can be cropped.')
            raise RuntimeError

        return samples


    def landmarks_detector(self, video_frames):
        landmarks = self.detect(video_frames, self.full_range_detector)
        if all(element is None for element in landmarks):
            landmarks = self.detect(video_frames, self.short_range_detector)
            assert any(l is not None for l in landmarks), "Cannot detect any frames in the video"
        return landmarks

    def video_process(self, video, landmarks):
        # Pre-process landmarks: interpolate frames that are not detected
        preprocessed_landmarks = self.interpolate_landmarks(landmarks)
        # Exclude corner cases: no landmark in all frames
        if not preprocessed_landmarks:
            return
        # Affine transformation and crop patch
        sequence = self.crop_patch(video, preprocessed_landmarks)
        assert sequence is not None, "crop an empty patch."
        return sequence

    def load_video(self, data_filename):
        return torchvision.io.read_video(data_filename, pts_unit="sec")[0].numpy()

    def detect(self, video_frames, detector):
        landmarks = []
        for frame in video_frames:
            results = detector.process(frame)
            if not results.detections:
                landmarks.append(None)
                continue
            face_points = []
            for idx, detected_faces in enumerate(results.detections):
                max_id, max_size = 0, 0
                bboxC = detected_faces.location_data.relative_bounding_box
                ih, iw, ic = frame.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                bbox_size = (bbox[2] - bbox[0]) + (bbox[3] - bbox[1])
                if bbox_size > max_size:
                    max_id, max_size = idx, bbox_size
                lmx = [
                    [int(detected_faces.location_data.relative_keypoints[
                             self.mp_face_detection.FaceKeyPoint(0).value].x * iw),
                     int(detected_faces.location_data.relative_keypoints[
                             self.mp_face_detection.FaceKeyPoint(0).value].y * ih)],
                    [int(detected_faces.location_data.relative_keypoints[
                             self.mp_face_detection.FaceKeyPoint(1).value].x * iw),
                     int(detected_faces.location_data.relative_keypoints[
                             self.mp_face_detection.FaceKeyPoint(1).value].y * ih)],
                    [int(detected_faces.location_data.relative_keypoints[
                             self.mp_face_detection.FaceKeyPoint(2).value].x * iw),
                     int(detected_faces.location_data.relative_keypoints[
                             self.mp_face_detection.FaceKeyPoint(2).value].y * ih)],
                    [int(detected_faces.location_data.relative_keypoints[
                             self.mp_face_detection.FaceKeyPoint(3).value].x * iw),
                     int(detected_faces.location_data.relative_keypoints[
                             self.mp_face_detection.FaceKeyPoint(3).value].y * ih)],
                ]
                face_points.append(lmx)
            landmarks.append(np.array(face_points[max_id]))
        return landmarks

    def crop_patch(self, video, landmarks):
        sequence = []
        for frame_idx, frame in enumerate(video):
            window_margin = min(
                self.window_margin // 2, frame_idx, len(landmarks) - 1 - frame_idx
            )
            smoothed_landmarks = np.mean(
                [
                    landmarks[x]
                    for x in range(
                    frame_idx - window_margin, frame_idx + window_margin + 1
                )
                ],
                axis=0,
            )
            smoothed_landmarks += landmarks[frame_idx].mean(
                axis=0
            ) - smoothed_landmarks.mean(axis=0)
            transformed_frame, transformed_landmarks = self.affine_transform(
                frame, smoothed_landmarks, self.reference
            )

            patch = self.cut_patch(
                transformed_frame,
                transformed_landmarks[self.start_idx: self.stop_idx],
                self.crop_height // 2,
                self.crop_width // 2,
            )
            sequence.append(patch)
        return np.array(sequence)

    def interpolate_landmarks(self, landmarks):
        valid_frames_idx = [idx for idx, lm in enumerate(landmarks) if lm is not None]

        if not valid_frames_idx:
            return None

        for idx in range(1, len(valid_frames_idx)):
            if valid_frames_idx[idx] - valid_frames_idx[idx - 1] > 1:
                landmarks = self.linear_interpolate(
                    landmarks, valid_frames_idx[idx - 1], valid_frames_idx[idx]
                )

        valid_frames_idx = [idx for idx, lm in enumerate(landmarks) if lm is not None]

        # Handle corner case: keep frames at the beginning or at the end that failed to be detected
        if valid_frames_idx:
            landmarks[: valid_frames_idx[0]] = [
                                                   landmarks[valid_frames_idx[0]]
                                               ] * valid_frames_idx[0]
            landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (
                    len(landmarks) - valid_frames_idx[-1]
            )

        assert all(lm is not None for lm in landmarks), "not every frame has landmark"

        return landmarks

    def affine_transform(
            self,
            frame,
            landmarks,
            reference,
            target_size=(256, 256),
            reference_size=(256, 256),
            stable_points=(0, 1, 2, 3),
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            border_value=0,
    ):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        stable_reference = self.get_stable_reference(
            reference, reference_size, target_size
        )
        transform = self.estimate_affine_transform(
            landmarks, stable_points, stable_reference
        )
        transformed_frame, transformed_landmarks = self.apply_affine_transform(
            frame,
            landmarks,
            transform,
            target_size,
            interpolation,
            border_mode,
            border_value,
        )

        return transformed_frame, transformed_landmarks

    def get_stable_reference(self, reference, reference_size, target_size):
        # -- right eye, left eye, nose tip, mouth center
        stable_reference = np.vstack(
            [
                np.mean(reference[36:42], axis=0),
                np.mean(reference[42:48], axis=0),
                np.mean(reference[31:36], axis=0),
                np.mean(reference[48:68], axis=0),
            ]
        )
        stable_reference[:, 0] -= (reference_size[0] - target_size[0]) / 2.0
        stable_reference[:, 1] -= (reference_size[1] - target_size[1]) / 2.0
        return stable_reference

    def estimate_affine_transform(self, landmarks, stable_points, stable_reference):
        return cv2.estimateAffinePartial2D(
            np.vstack([landmarks[x] for x in stable_points]),
            stable_reference,
            method=cv2.LMEDS,
        )[0]

    def apply_affine_transform(
            self,
            frame,
            landmarks,
            transform,
            target_size,
            interpolation,
            border_mode,
            border_value,
    ):
        transformed_frame = cv2.warpAffine(
            frame,
            transform,
            dsize=(target_size[0], target_size[1]),
            flags=interpolation,
            borderMode=border_mode,
            borderValue=border_value,
        )
        transformed_landmarks = (
                np.matmul(landmarks, transform[:, :2].transpose())
                + transform[:, 2].transpose()
        )
        return transformed_frame, transformed_landmarks

    def linear_interpolate(self, landmarks, start_idx, stop_idx):
        start_landmarks = landmarks[start_idx]
        stop_landmarks = landmarks[stop_idx]
        delta = stop_landmarks - start_landmarks
        for idx in range(1, stop_idx - start_idx):
            landmarks[start_idx + idx] = (
                    start_landmarks + idx / float(stop_idx - start_idx) * delta
            )
        return landmarks

    def cut_patch(self, img, landmarks, height, width, threshold=5):
        center_x, center_y = np.mean(landmarks, axis=0)
        # Check for too much bias in height and width
        if abs(center_y - img.shape[0] / 2) > height + threshold:
            raise OverflowError("too much bias in height")
        if abs(center_x - img.shape[1] / 2) > width + threshold:
            raise OverflowError("too much bias in width")
        # Calculate bounding box coordinates
        y_min = int(round(np.clip(center_y - height, 0, img.shape[0])))
        y_max = int(round(np.clip(center_y + height, 0, img.shape[0])))
        x_min = int(round(np.clip(center_x - width, 0, img.shape[1])))
        x_max = int(round(np.clip(center_x + width, 0, img.shape[1])))
        # Cut the image
        cutted_img = np.copy(img[y_min:y_max, x_min:x_max])

        return cutted_img