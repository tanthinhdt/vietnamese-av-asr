import cv2
import torch
import numpy as np
from pathlib import Path
import torch.nn.functional as F
import torchaudio.transforms as TA
import torchvision.transforms.v2 as TV
from typing import Dict, Any, Tuple
from transformers import Pipeline
from python_speech_features import logfbank
from optimum.onnxruntime import ORTModelForCausalLM
from mediapipe.python.solutions.face_detection import FaceDetection, FaceKeyPoint
from loguru import logger


class AutomaticSpeechRecognitionPipeline(Pipeline):
    INSTRUCTIONS = {
        "vi": "Hãy nhận diện câu tiếng Việt này. Đầu vào: ",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.assistant_model = kwargs.pop("assistant_model", None)
        if self.assistant_model is not None:
            model_kwargs = kwargs.get("model_kwargs", {})
            self.assistant_model = ORTModelForCausalLM.from_pretrained(
                self.assistant_model,
                trust_remote_code=True,
                cache_dir=model_kwargs.get("cache_dir", None),
            )

        crop_size = (self.feature_extractor.height, self.feature_extractor.width)
        self.transforms = TV.Compose(
            [
                Sanitize(),
                Extract(
                    crop_size=crop_size,
                    mean=self.feature_extractor.mean,
                    std=self.feature_extractor.std,
                    sampling_rate=self.feature_extractor.sampling_rate,
                    stack_order=self.feature_extractor.stack_order,
                    eos_token_id=self.tokenizer.eos_token_id,
                    normalization=self.feature_extractor.normalization,
                    label_rate=self.feature_extractor.label_rate,
                ),
                Collate(),
                Pad(
                    max_sample_size=self.feature_extractor.max_sample_size,
                    pad_audio=self.feature_extractor.pad_audio
                ),
                ToDevice(device=self.device),
            ]
        )

    def _sanitize_parameters(self, **kwargs):
        # Sanitize the parameters for preprocessing
        preprocess_kwargs = {}
        preprocess_kwargs["lang"] = kwargs.pop("lang", "vi")
        assert preprocess_kwargs["lang"] in self.INSTRUCTIONS, \
            f"Language {preprocess_kwargs['lang']} is not supported"
        # Sanitize the parameters for the forward pass
        forward_kwargs = {}
        # Sanitize the parameters for postprocessing
        postprocess_kwargs = kwargs

        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def preprocess(
        self,
        inputs: Dict[str, Dict[str, Any]],
        lang: str = "vi",
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocesses the inputs to the model.

        Parameters
        ----------
        inputs : Dict[str, Union[np.ndarray, torch.Tensor]]
            The inputs to the model:
                "video": dict,
                    "fps": float or int,
                    "data": np.ndarray or torch.Tensor
                        (time, height, width, channels).
                "audio":
                    "sampling_rate": int,
                    "data": np.ndarray or torch.Tensor
                        (channels, time).

        Returns
        -------
        Dict[str, torch.Tensor]
            The preprocessed inputs:
                "video": torch.Tensor
                    (batch, channels, time, height, width).
                "audio": torch.Tensor
                    (batch, time * stack_order * channels).
        """
        tokenizer_output = self.tokenizer(
            self.INSTRUCTIONS[lang],
            return_tensors="pt",
        )
        inputs["text_source"] = tokenizer_output.input_ids[0]
        logger.info("Tokenized instruction")

        return self.transforms(inputs)

    def _forward(self, inputs: torch.Tensor) -> torch.Tensor:
        llm_input, _ = self.model.embed(**inputs)
        return llm_input

    def postprocess(self, llm_input: torch.Tensor, **kwargs) -> str:
        if self.assistant_model is None:
            self.model.decoder.config.use_cache = True
            return self.model.decoder.generate(inputs_embeds=llm_input, **kwargs)
        return self.assistant_model.generate(inputs_embeds=llm_input, **kwargs)


class Sanitize:
    def __sanitize_video(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if inputs is not None:
            assert "data" in inputs, \
                f"Expected video input to have 'data' key, got {inputs.keys()}"
            assert "fps" in inputs, \
                f"Expected video input to have 'fps' key, got {inputs.keys()}"

            data_shape = inputs["data"].shape
            assert len(data_shape) == 4, \
                f"Expected video input to have 5 dimensions, got {data_shape}"
            assert data_shape[-1] == 3, \
                f"Expected video input to have 3 channels, got {data_shape[-1]}"

        return inputs

    def __sanitize_audio(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if inputs is not None:
            assert "data" in inputs, \
                f"Expected video input to have 'data' key, got {inputs.keys()}"
            assert "sampling_rate" in inputs, \
                f"Expected video input to have 'sampling_rate' key, got {inputs.keys()}"
        return inputs

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        inputs["video"] = self.__sanitize_video(inputs.get("video", None))
        inputs["audio"] = self.__sanitize_audio(inputs.get("audio", None))

        for key, value in inputs.items():
            if value is None or isinstance(value, torch.Tensor):
                continue
            if isinstance(value["data"], np.ndarray):
                inputs[key]["data"] = torch.from_numpy(value["data"])
            else:
                inputs[key]["data"] = value["data"].cpu()

        logger.info("Sanitized inputs")
        return inputs


class CropMouth:
    def __init__(self, crop_size: tuple) -> None:
        self.crop_height, self.crop_width = crop_size

        self.short_range_detector = FaceDetection(
            min_detection_confidence=0.5,
            model_selection=0,
        )
        self.full_range_detector = FaceDetection(
            min_detection_confidence=0.5,
            model_selection=1,
        )
        self.reference = np.load(Path(__file__).parent / "mean_face.npy")
        self.start_idx = 3
        self.stop_idx = 4
        self.window_margin = 12

    def __call__(self, video_frames: torch.Tensor):
        landmarks = self.landmarks_detector(video_frames.numpy())
        video = self.video_process(video_frames.numpy(), landmarks)
        video = torch.from_numpy(video) if video is not None else None

        logger.info("Cropped mouth")
        return video

    def landmarks_detector(self, video_frames):
        landmarks = self.detect(video_frames, self.full_range_detector)
        if all(element is None for element in landmarks):
            landmarks = self.detect(video_frames, self.short_range_detector)
            assert any(landmark is not None for landmark in landmarks), \
                "Cannot detect any frames in the video"
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

    def detect(
        self,
        video_frames: np.ndarray,
        detector: FaceDetection,
    ) -> np.ndarray:
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
                bbox = (
                    int(bboxC.xmin * iw),
                    int(bboxC.ymin * ih),
                    int(bboxC.width * iw),
                    int(bboxC.height * ih),
                )
                bbox_size = (bbox[2] - bbox[0]) + (bbox[3] - bbox[1])
                if bbox_size > max_size:
                    max_id, max_size = idx, bbox_size

                keypoints = detected_faces.location_data.relative_keypoints
                lmx = [
                    [int(keypoints[FaceKeyPoint(0).value].x * iw),
                     int(keypoints[FaceKeyPoint(0).value].y * ih)],
                    [int(keypoints[FaceKeyPoint(1).value].x * iw),
                     int(keypoints[FaceKeyPoint(1).value].y * ih)],
                    [int(keypoints[FaceKeyPoint(2).value].x * iw),
                     int(keypoints[FaceKeyPoint(2).value].y * ih)],
                    [int(keypoints[FaceKeyPoint(3).value].x * iw),
                     int(keypoints[FaceKeyPoint(3).value].y * ih)],
                    ]
                face_points.append(lmx)

            landmarks.append(np.array(face_points[max_id]))
        return landmarks

    def crop_patch(
        self,
        video: np.ndarray,
        landmarks: np.ndarray,
    ) -> np.ndarray:
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

    def interpolate_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
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
        frame: np.ndarray,
        landmarks: np.ndarray,
        reference: np.ndarray,
        target_size: tuple = (256, 256),
        reference_size: tuple = (256, 256),
        stable_points: tuple = (0, 1, 2, 3),
        interpolation: int = cv2.INTER_LINEAR,
        border_mode: int = cv2.BORDER_CONSTANT,
        border_value: int = 0,
    ) -> tuple:
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

    def get_stable_reference(
        self,
        reference: np.ndarray,
        reference_size: tuple,
        target_size: tuple,
    ) -> np.ndarray:
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
        frame: np.ndarray,
        landmarks: np.ndarray,
        transform: np.ndarray,
        target_size: tuple,
        interpolation: int,
        border_mode: int,
        border_value: int,
    ) -> tuple:
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

    def linear_interpolate(
        self,
        landmarks: np.ndarray,
        start_idx: int,
        stop_idx: int,
    ) -> np.ndarray:
        start_landmarks = landmarks[start_idx]
        stop_landmarks = landmarks[stop_idx]
        delta = stop_landmarks - start_landmarks
        for idx in range(1, stop_idx - start_idx):
            landmarks[start_idx + idx] = (
                    start_landmarks + idx / float(stop_idx - start_idx) * delta
            )
        return landmarks

    def cut_patch(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        height: int,
        width: int,
        threshold: int = 5,
    ) -> np.ndarray:
        center_x, center_y = np.mean(landmarks, axis=0)
        # Check for too much bias in height and width
        if abs(center_y - frame.shape[0] / 2) > height + threshold:
            raise OverflowError("too much bias in height")
        if abs(center_x - frame.shape[1] / 2) > width + threshold:
            raise OverflowError("too much bias in width")
        # Calculate bounding box coordinates
        y_min = int(round(np.clip(center_y - height, 0, frame.shape[0])))
        y_max = int(round(np.clip(center_y + height, 0, frame.shape[0])))
        x_min = int(round(np.clip(center_x - width, 0, frame.shape[1])))
        x_max = int(round(np.clip(center_x + width, 0, frame.shape[1])))
        # Cut the image
        cutted_img = np.copy(frame[y_min:y_max, x_min:x_max])
        return cutted_img


class ConvertStereoToMono:
    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            The input tensor with shape (channels, time).

        Returns
        -------
        torch.Tensor
            The output tensor with shape (1, time).
        """
        if inputs["data"].shape[0] > 1:
            inputs["data"] = torch.mean(inputs["data"], dim=0, keepdim=True)

        logger.info("Converted stereo to mono")
        return inputs


class Resample:
    def __init__(self, sampling_rate: int) -> None:
        self.sampling_rate = sampling_rate

    def __call__(self, inputs: Dict[str, Any]) -> torch.Tensor:
        data = inputs["data"]
        sampling_rate = inputs["sampling_rate"]

        if sampling_rate != self.sampling_rate:
            transform = TA.Resample(
                orig_freq=sampling_rate,
                new_freq=self.sampling_rate,
            )
            data = transform(data)

        logger.info("Resampled audio")
        return data


class LogMelFilterBank:
    def __init__(self, sampling_rate: int) -> None:
        self.sampling_rate = sampling_rate

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = torch.from_numpy(
            logfbank(
                inputs.numpy(),
                samplerate=self.sampling_rate,
            )
        )
        inputs = inputs.to(torch.float32)

        logger.info("Computed log mel filter bank")
        return inputs


class Stack:
    def __init__(self, stack_order: int) -> None:
        self.stack_order = stack_order

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        feat_dim = inputs.shape[1]

        if len(inputs) % self.stack_order != 0:
            res = self.stack_order - len(inputs) % self.stack_order
            res = torch.zeros([res, feat_dim], dtype=inputs.dtype)
            inputs = torch.concatenate([inputs, res], dim=0)
        inputs = (
            inputs
            .reshape((-1, self.stack_order, feat_dim))
            .reshape(-1, self.stack_order * feat_dim)
        )

        logger.info("Stacked audio")
        return inputs


class Normalize:
    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            inputs = F.layer_norm(inputs, inputs.shape[1:])

        logger.info("Normalized audio")
        return inputs


class Tokenize:
    def __init__(
        self,
        eos_token_id: int,
        label_rate: int = -1,
    ) -> None:
        self.eos_token_id = eos_token_id
        self.label_rate = label_rate

    def __call__(
        self,
        values: torch.Tensor,
        left_pad: bool = False,
        move_eos_to_beginning: bool = False,
        pad_to_length: int = None,
        pad_to_multiple: int = 1,
    ) -> torch.Tensor:

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            if move_eos_to_beginning:
                if eos_idx is None:
                    # if no eos_idx is specified, then use the last token in src
                    dst[0] = src[-1]
                else:
                    dst[0] = eos_idx
                dst[1:] = src[:-1]
            else:
                dst.copy_(src)

        if self.label_rate != -1:
            raise NotImplementedError("not yet")

        pad_idx, eos_idx = 0, self.eos_token_id

        size = values.size(0)
        size = size if pad_to_length is None else max(size, pad_to_length)
        if pad_to_multiple != 1 and size % pad_to_multiple != 0:
            size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)

        res = values.new(size).fill_(pad_idx)
        copy_tensor(values, res[size - len(values):] if left_pad else res[: len(values)])

        logger.info("Tokenized text")
        return res


class Extract:
    def __init__(
        self,
        crop_size: tuple,
        mean: tuple,
        std: tuple,
        sampling_rate: int,
        stack_order: int,
        eos_token_id: int,
        normalization: bool = True,
        label_rate: int = -1,
    ) -> None:
        self.video_transforms = TV.Compose(
            [
                CropMouth(crop_size),
                TV.ToDtype(torch.float32),
                TV.Lambda(lambda x: x.permute(0, 3, 1, 2)),
                TV.Grayscale(num_output_channels=1),
                TV.Normalize((0.0,), (255.0,)),
                TV.Normalize((mean,), (std,)),
                TV.ToDtype(torch.float32),
            ]
        )

        self.audio_transforms = [
            ConvertStereoToMono(),
            Resample(sampling_rate),
            LogMelFilterBank(sampling_rate),
            Stack(stack_order),
            TV.ToDtype(torch.float32),
        ]
        if normalization:
            self.audio_transforms.append(Normalize())
        self.audio_transforms = TV.Compose(self.audio_transforms)

        self.text_transforms = TV.Compose(
            [
                Tokenize(eos_token_id, label_rate)
            ]
        )

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if inputs["video"] is not None:
            inputs["video"] = self.video_transforms(inputs["video"]["data"])
        if inputs["audio"] is not None:
            inputs["audio"] = self.audio_transforms(inputs["audio"])
        tokens = self.text_transforms(inputs["text_source"])

        logger.info("Extracted features")
        return {
            "source": {
                "audio": inputs["audio"],
                "video": inputs["video"],
                "text": tokens,
            },
        }


class Collate:
    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        inputs = {
            "source": {
                "audio": torch.stack([inputs["source"]["audio"]]),
                "video": torch.stack([inputs["source"]["video"]]),
                "text": torch.stack([inputs["source"]["text"]]),
            },
        }

        logger.info("Collated inputs")
        return inputs


class Pad:
    def __init__(self, max_sample_size: int, pad_audio: bool) -> None:
        self.max_sample_size = max_sample_size
        self.pad_audio = pad_audio

    def __crop_to_max_size(
        self,
        inputs: torch.Tensor,
        target_size: int,
        start: int = None,
    ) -> Tuple[torch.Tensor, int]:
        size = len(inputs)
        diff = size - target_size
        if diff <= 0:
            return inputs, 0
        # longer utterances
        if start is None:
            start, end = 0, target_size
            if self.random_crop:
                start = np.random.randint(0, diff + 1)
                end = size - diff + start
        else:
            end = start + target_size
        return inputs[start:end], start

    def __generate_padding_mask(
        self,
        inputs: torch.Tensor,
        target_size: int,
        start: int = None,
    ) -> Tuple[torch.Tensor, int, int]:
        padding_mask = torch.BoolTensor(target_size).fill_(False)
        start = start or None

        diff = len(inputs) - target_size
        if diff < 0:
            assert self.pad_audio, "Cannot pad audio"
            inputs = torch.cat(
                [
                    inputs,
                    inputs.new_full([-diff] + list(inputs.shape[1:]), 0.0),
                ]
            )
            padding_mask[diff:] = True
        else:
            inputs, start = self.__crop_to_max_size(inputs, target_size, start)

        if len(inputs.shape) == 3:
            # [B, T, F] -> [B, F, T]
            inputs = inputs.transpose(1, 2)
        else:
            # [B, T, C, H, W] -> [B, C, T, H, W]
            inputs = inputs.permute((0, 2, 1, 3, 4)).contiguous()

        return inputs, padding_mask, start

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        video = inputs["source"]["video"]
        audio = inputs["source"]["audio"]

        if video is not None and audio is not None:
            diff = len(audio) - len(video)
            if diff < 0:
                audio = torch.concatenate(
                    [
                        audio,
                        torch.zeros(
                            [-diff, audio.shape[-1]],
                            dtype=audio.dtype
                        ),
                    ]
                )
            elif diff > 0:
                audio = audio[:-diff]

        if audio is not None:
            audio_size = len(audio)
        else:
            audio_size = len(video)

        if self.pad_audio:
            audio_size = min(audio_size, self.max_sample_size)
        else:
            audio_size = max(audio_size, self.max_sample_size)

        if audio is not None:
            audio, audio_mask, audio_start = self.__generate_padding_mask(
                audio, audio_size
            )

        if video is not None:
            video, video_mask, video_start = self.__generate_padding_mask(
                video, audio_size
            )

        assert audio_size == len(audio) == len(video), \
            "Audio and video sizes are not equal"
        assert audio_mask == video_mask, "Audio and video masks are not equal"
        assert audio_start == video_start, "Audio and video starts are not equal"

        return {
            "source": {
                "audio": audio,
                "video": video,
                "text": inputs["text"],
            },
            "padding_mask": audio_mask,
        }


class ToDevice:
    def __init__(self, device: torch.device) -> None:
        self.device = device

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        inputs = {
            "source": {
                "audio": inputs["source"]["audio"].to(self.device),
                "video": inputs["source"]["video"].to(self.device),
                "text": inputs["source"]["text"].to(self.device),
            },
            "padding_mask": inputs["padding_mask"].to(self.device),
        }

        logger.info("Moved inputs to device")
        return inputs
