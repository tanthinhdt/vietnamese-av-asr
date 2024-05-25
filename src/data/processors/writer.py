import os

import cv2
import numpy as np
import torch
import torchaudio

from vlr.data.processors import Processor


class Writer(Processor):
    """
    This class is used to save data to disk.
    """
    def __init__(self) -> None:
        """
        Args:
            file_format: File format of output.
        """
        self.operations = {
            (np.ndarray, "mp4"): self.numpy_to_mp4,
        }

    def process(
        self, sample: dict,
        schemas: dict,
    ) -> dict:
        """
        Save data of each schema to disk.

        Args:
            sample: Sample.
            schemas: Dictionary where key is schema name and value is tuple of
                output directory, file format and keyword arguments for saving function.

        Returns:
            Original sample.
        """
        for schema, (output_dir, file_format, kwags) in schemas.items():
            data = sample[schema]
            file_path = os.path.join(output_dir, sample["id"] + "." + file_format)
            self.operations[(type(data), file_format)](data, file_path, **kwags)
        return sample

    def numpy_to_mp4(self, array: np.ndarray, file_path: str, fps: int) -> None:
        """
        Save numpy array to mp4.

        Args:
            array: Numpy array.
            file_path: File path.
            fps: Frame rate.
        """
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width, height, _ = array[0].shape
        writer = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
        for frame in array:
            writer.write(frame)
        writer.release()

    def numpy_to_wav(
        self, array: np.ndarray,
        file_path: str,
        sampling_rate: int
    ) -> None:
        """
        Save numpy array to wav.

        Args:
            array: Numpy array.
            file_path: File path.
            sampling_rate: Sampling rate.
        """
        torchaudio.save(
            uri=file_path,
            src=torch.from_numpy(array),
            sample_rate=sampling_rate,
            channels_first=True,
        )
