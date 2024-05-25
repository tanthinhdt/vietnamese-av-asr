# Copyright 2023 Thinh T. Duong
import os
import datasets
from huggingface_hub import HfFileSystem
from typing import List, Tuple


logger = datasets.logging.get_logger(__name__)
fs = HfFileSystem()


_CITATION = """

"""
_DESCRIPTION = """
    This dataset contain denoised audio of Vietnamese speakers.
"""
_HOMEPAGE = "https://github.com/tanthinhdt/vietnamese-av-asr"
_REPO_PATH = "datasets/phdkhanh2507/denoised-vietnamese-audio"
_REPO_URL = f"https://huggingface.co/{_REPO_PATH}/resolve/main"
_URLS = {
    "meta": f"{_REPO_URL}/metadata/" + "{channel}.parquet",
    "audio": f"{_REPO_URL}/audio/" + "{channel}.zip",
}
_CONFIGS = ["all"]
if fs.exists(_REPO_PATH + "/metadata"):
    _CONFIGS.extend([
        os.path.basename(file_name)[:-8]
        for file_name in fs.listdir(_REPO_PATH + "/metadata", detail=False)
        if file_name.endswith(".parquet")
    ])


class DenoisedVietnameseAudioConfig(datasets.BuilderConfig):
    """Denoised Vietnamese Audio configuration."""

    def __init__(self, name, **kwargs):
        """
        :param name:    Name of subset.
        :param kwargs:  Arguments.
        """
        super().__init__(
            name=name,
            version=datasets.Version("1.0.0"),
            description=_DESCRIPTION,
            **kwargs,
        )


class DenoisedVietnameseAudio(datasets.GeneratorBasedBuilder):
    """Denoised Vietnamese Audio dataset."""

    BUILDER_CONFIGS = [DenoisedVietnameseAudioConfig(name) for name in _CONFIGS]
    DEFAULT_CONFIG_NAME = "all"

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features({
            "id": datasets.Value("string"),
            "channel": datasets.Value("string"),
            "duration": datasets.Value("float64"),
            "fps": datasets.Value("int8"),
            "audio": datasets.Value("string"),
            "sampling_rate": datasets.Value("int64"),
        })

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> List[datasets.SplitGenerator]:
        """
        Get splits.
        :param dl_manager:  Download manager.
        :return:            Splits.
        """
        config_names = _CONFIGS[1:] if self.config.name == "all" else [self.config.name]

        metadata_paths = dl_manager.download(
            [_URLS["meta"].format(channel=channel) for channel in config_names]
        )
        audio_dirs = dl_manager.download_and_extract(
            [_URLS["audio"].format(channel=channel) for channel in config_names]
        )

        audio_dict = {
            channel: audio_dir for channel, audio_dir in zip(config_names, audio_dirs)
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "metadata_paths": metadata_paths,
                    "audio_dict": audio_dict,
                },
            ),
        ]

    def _generate_examples(
        self, metadata_paths: List[str],
        audio_dict: dict,
    ) -> Tuple[int, dict]:
        """
        Generate examples from metadata.
        :param metadata_paths:      Paths to metadata.
        :param audio_dict:          Paths to directory containing audio.
        :yield:                     Example.
        """
        dataset = datasets.load_dataset(
            "parquet",
            data_files=metadata_paths,
            split="train",
        )
        for i, sample in enumerate(dataset):
            channel = sample["channel"]
            audio_path = os.path.join(
                audio_dict[channel], channel, sample["id"] + ".wav"
            )

            yield i, {
                "id": sample["id"],
                "channel": channel,
                "duration": sample["duration"],
                "fps": sample["fps"],
                "audio": audio_path,
                "sampling_rate": sample["sampling_rate"],
            }
print(os.path.split('/Users/minhnguyen/Downloads/AIP_Project/phdkhanh2507/testVLR/.cache/test/downloads/22bec53d8b7cb8c5c0272ea050ff4acddfc802d119bb8b5248d338a6b2b5e5f4'))