import os
import torch
import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio
from .processor import Processor


class Denoiser(Processor):
    """
    This class is used to denoise audio array.
    """
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = pretrained.dns64().to(self.device)

    def process(
        self, sample: dict,
        audio_output_dir: str,
        output_sampling_rate: int = 16000,
    ) -> dict:
        """
        Denoise audio array.
        :param sample:                  Sample.
        :param audio_output_dir:        Path to directory containing denoised audio array.
        :param output_sampling_rate:    Sampling rate of denoised audio array.
        :return:                        Sample updated with path to denoised audio array.
        """
        audio_output_path = os.path.join(audio_output_dir, sample["id"][0] + ".wav")

        if not os.path.exists(audio_output_path):
            try:
                audio_array, sampling_rate = torchaudio.load(sample["audio"][0])
                audio_array = convert_audio(
                    audio_array.to(self.device),
                    sampling_rate,
                    self.model.sample_rate,
                    self.model.chin
                )

                with torch.no_grad():
                    denoised_audio_array = self.model(audio_array[None].float())[0].cpu()

                torchaudio.save(
                    audio_output_path,
                    denoised_audio_array,
                    output_sampling_rate,
                )
            except Exception:
                sample["id"] = [None]

        sample["sampling_rate"][0] = output_sampling_rate
        return sample
