import av
import numpy as np
from io import BytesIO
from requests import get
from typing import Union
from zipfile import ZipFile
from huggingface_hub import HfFolder
from vlr.data.processors import Processor


class Reader(Processor):
    """
    This class is used to read and convert data from different sources.
    """
    TOKEN = HfFolder.get_token()

    def read_video_from_bytes(
        self, video_bytes: bytes,
        include_audio: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Read video from zip directory.
        :param zip_dir_obj:     Zip directory object.
        :param path_in_zip:     Path to file in zip directory.
        :return:                Video in bytes.
        """
        visual, audio, metadata = self._bytesio_to_arrays(
            bytesio_obj=BytesIO(video_bytes)
        )
        assert visual is not None, "Cannot read video."
        if include_audio:
            assert audio is not None, "Cannot read audio."
        return visual, audio, metadata

    def _bytesio_to_arrays(
        self, bytesio_obj: BytesIO,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Decode video into frames and audio into array.
        :param bytesio_obj:     BytesIO object.
        :return:                Tuple of video array, audio array and metadata.
        """
        container = av.open(bytesio_obj)

        # Find the video stream and audio stream
        visual_stream = next(iter(container.streams.video), None)
        audio_stream = next(iter(container.streams.audio), None)

        # Extract metadata
        metadata = {
            "visual_fps": visual_stream.base_rate,
            "audio_fps": audio_stream.rate,
        }

        # Iterate over packets and extract frames and audio
        visual, audio = None, None
        for packet in container.demux([visual_stream, audio_stream]):
            if packet.stream.type == 'video':
                visual = self._visual_packet_to_array(packet)
            elif packet.stream.type == 'audio':
                audio = self._audio_packet_to_array(packet)

        return visual, audio, metadata

    def _visual_packet_to_array(self, packet: av.Packet) -> np.ndarray:
        """
        Convert visual packet to array.
        :param packet:  Visual packet.
        :return:        Visual array with shape (time, height, width, channel).
        """
        assert packet.stream.type == "video"
        frames = []
        for frame in packet.decode():
            img_array = np.array(frame.to_image())
            frames.append(img_array)
        return np.array(frames) if len(frames) > 0 else None

    def _audio_packet_to_array(self, packet: av.Packet) -> np.ndarray:
        """
        Convert audio packet to array.
        :param packet:  Audio packet.
        :return:        Audio array with shape (time, channel).
        """
        assert packet.stream.type == "audio"
        samples = []
        for frame in packet.decode():
            samples.append(frame.to_ndarray())
        return np.concatenate(samples, axis=1) if len(samples) > 0 else None
