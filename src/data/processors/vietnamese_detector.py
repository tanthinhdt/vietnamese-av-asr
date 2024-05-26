import shutil

import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from .processor import Processor
from typing import Tuple


class VietnameseDetector(Processor):
    """
    This class is used to filter out samples with Vietnamese language.
    """
    def __init__(self) -> None:
        """
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/lang-id-voxlingua107-ecapa",
            savedir="tmp"
        ).to(device=self.device)
        self.sampling_rate = 16000

    def classify(self, audio_array: torch.Tensor, sampling_rate: int) -> Tuple[int, float]:
        """
        Classify language of audio array.
        :param audio_array:     Audio array.
        :return:                Language index and score.
        """
        if sampling_rate != self.sampling_rate:
            audio_array = torchaudio.transforms.Resample(
                orig_freq=sampling_rate,
                new_freq=self.sampling_rate,
            )

        _, score, lang_idx, _ = self.model.classify_batch(audio_array.to(self.device))
        score = score.exp().item()
        lang_idx = lang_idx.item()

        torch.cuda.empty_cache()
        return lang_idx, score

    def is_vietnamese(
        self, audio_array: torch.Tensor,
        sampling_rate: int,
        threshold: float = 0.99,
    ) -> bool:
        """
        Check if language is Vietnamese.
        :param lang_idx:        Language index.
        :param score:           Score.
        :param threshold:       Threshold.
        :return:                Whether language is Vietnamese.
        """
        lang_idx, score = self.classify(audio_array, sampling_rate)
        return lang_idx == 102 and score >= threshold

    def process(
            self,
            sample: dict,
            visual_output_dir: str,
            audio_output_dir: str,
            *args,
            **kwargs) -> dict:
        visual_path = sample['visual_path'][0]
        audio_path = sample['audio_path'][0]
        is_vietnamese = self.is_vietnamese(
            *torchaudio.load(audio_path)
        )
        if is_vietnamese:
            shutil.copy(src=visual_path,dst=visual_output_dir)
            shutil.copy(src=audio_path, dst=audio_output_dir)
        else:
            sample['id'][0] = None
        return sample




