import torch
import numpy as np
import mediapipe as mp
import torch.nn.functional as F
import torchaudio.transforms as TA
import torchvision.transforms.v2 as TV
from typing import Dict, Any, Tuple
from transformers import Pipeline
from python_speech_features import logfbank
from optimum.onnxruntime import ORTModelForCausalLM


class AutomaticSpeechRecognitionPipeline(Pipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if kwargs.pop("use_onnx", False):
            model_kwargs = kwargs.pop("model_kwargs", {})
            self.assistant_model = ORTModelForCausalLM.from_pretrained(
                kwargs.pop("assistant_model", "vilm/vinallama-2.7b"),
                trust_remote_code=True,
                cache_dir=model_kwargs.pop("cache_dir", None),
            )
        else:
            self.assistant_model = None

        self.instructions = {
            "vi": "Hãy nhận diện câu tiếng Việt này. Đầu vào: ",
        }

        crop_size = (
            self.feature_extractor.size["height"],
            self.feature_extractor.size["width"]
        )
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
        assert preprocess_kwargs["lang"] in self.instructions, \
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
                        (time, channels).

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
            self.instructions[lang],
            return_tensors="pt",
        )
        inputs["text_source"] = tokenizer_output.input_ids[0]
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
                f"Expected video input to have 'fps' key, got {inputs.keys()}"

            data_shape = inputs["data"].shape
            assert len(data_shape.shape) == 1, \
                f"Expected audio input to have 1 dimensions, got {data_shape.shape}"

        return inputs

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        inputs["video"] = self.__sanitize_video(inputs.get("video", None))
        inputs["audio"] = self.__sanitize_audio(inputs.get("audio", None))

        for key, value in inputs.items():
            if value is None:
                continue
            if isinstance(value[key]["data"], np.ndarray):
                value[key]["data"] = torch.from_numpy(value[key]["data"])
            else:
                value[key]["data"] = value[key]["data"].cpu()

        return inputs


class CropMouth:
    def __init__(self, crop_size: tuple) -> None:
        self.crop_size = crop_size
        self.landmark_detector = mp.solutions.face_mesh.FaceMesh()
        self.mouth_landmark_idxes = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        ]

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        mouths = []
        for frame in inputs:
            face_landmarks = self.landmark_detector.process(frame).multi_face_landmarks
            if face_landmarks is None:
                continue
            mouth_landmarks = np.array([
                [landmark.x, landmark.y] for landmark in face_landmarks[0].landmark
            ])[self.mouth_landmark_idxes, :]
            center_x = np.mean(mouth_landmarks[:, 0]) * frame.shape[1]
            min_x = int(center_x - self.crop_size / 2)
            max_x = int(center_x + self.crop_size / 2)
            center_y = np.mean(mouth_landmarks[:, 1]) * frame.shape[0]
            min_y = int(center_y - self.crop_size / 2)
            max_y = int(center_y + self.crop_size / 2)
            mouths.append(frame[min_y:max_y, min_x:max_x])
        return torch.stack(mouths)


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
        return data


class LogMelFilterBank:
    def __init__(self, sampling_rate: int) -> None:
        self.sampling_rate = sampling_rate

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = torch.from_numpy(logfbank(inputs.numpy(), samplerate=self.sampling_rate))
        return inputs.to(torch.float32)


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
        return inputs


class Normalize:
    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            inputs = F.layer_norm(inputs, inputs.shape[1:])
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
                TV.Lambda(lambda x: x.permute(0, 3, 1, 2)),
                TV.Grayscale(num_output_channels=1),
                TV.Normalize(0.0, 255.0),
                TV.CenterCrop(crop_size),
                TV.Normalize(mean, std),
                TV.ToDtype(torch.float32),
            ]
        )

        self.audio_transforms = [
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
        return {
            "source": {
                "audio": inputs["audio"],
                "video": inputs["video"],
                "text": tokens,
            },
        }


class Collate:
    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "source": {
                "audio": torch.stack([inputs["source"]["audio"]]),
                "video": torch.stack([inputs["source"]["video"]]),
                "text": torch.stack([inputs["source"]["text"]]),
            },
        }


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

        if len(inputs.shape) == 2:
            # [B, T, F] -> [B, F, T]
            inputs = inputs.transpose(1, 2)
        else:
            # [T, C, H, W] -> [C, T, H, W]
            inputs = inputs.permute((1, 0, 2, 3)).contiguous()

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
        return {
            "source": {
                "audio": inputs["source"]["audio"].to(self.device),
                "video": inputs["source"]["video"].to(self.device),
                "text": inputs["source"]["text"].to(self.device),
            },
            "padding_mask": inputs["padding_mask"].to(self.device),
        }
