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

"""
_HOMEPAGE = "https://github.com/tanthinhdt/vietnamese-av-asr"
_META_REPO_PATH = "datasets/phdkhanh2507/testVLR/builders/purified-vietnamese-audio"
_VISUAL_REPO_PATH = "datasets/phdkhanh2507/testVLR/builders/vietnamese-speaker-lip-clip"
_AUDIO_REPO_PATH = "datasets/phdkhanh2507/testVLR/builders/vietnamese-denoised-audio"
_REPO_URL = "https://huggingface.co/{}/resolve/main"
_URLS = {
    "meta": f"{_REPO_URL}/metadata/".format(_META_REPO_PATH) + "{channel}.parquet",
    "visual": f"{_REPO_URL}/visual/".format(_VISUAL_REPO_PATH) + "{channel}.zip",
    "audio": f"{_REPO_URL}/audio/".format(_AUDIO_REPO_PATH) + "{channel}.zip",
    "transcript": f"{_REPO_URL}/transcript/".format(_META_REPO_PATH) + "{channel}.zip",
}
_CONFIGS = ["all"]
if fs.exists(_META_REPO_PATH + "/metadata"):
    _CONFIGS.extend([
        os.path.basename(file_name)[:-8]
        for file_name in fs.listdir(_META_REPO_PATH + "/metadata", detail=False)
        if file_name.endswith(".parquet")
    ])


class testVLRConfig(datasets.BuilderConfig):
    """testVLR configuration."""

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


class testVLR(datasets.GeneratorBasedBuilder):
    """testVLR dataset."""

    BUILDER_CONFIGS = [testVLRConfig(name) for name in _CONFIGS]

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features({
            "id": datasets.Value("string"),
            "channel": datasets.Value("string"),
            "visual": datasets.Value("binary"),
            "duration": datasets.Value("float64"),
            "fps": datasets.Value("int8"),
            "audio": datasets.Value("binary"),
            "sampling_rate": datasets.Value("int64"),
            "transcript": datasets.Value("string"),
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
        dataset = datasets.load_dataset(
            "parquet",
            data_files=metadata_paths,
            split="train",
        )
        dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
        train_set = dataset["train"]
        val_test_set = dataset["test"].train_test_split(test_size=0.5)
        val_set = val_test_set["train"]
        test_set = val_test_set["test"]

        split_dict = {
            datasets.Split.TRAIN: train_set,
            datasets.Split.VALIDATION: val_set,
            datasets.Split.TEST: test_set,
        }

        visual_dirs = dl_manager.download_and_extract(
            [_URLS["visual"].format(channel=channel) for channel in config_names]
        )
        visual_dict = {
            channel: visual_dir
            for channel, visual_dir in zip(config_names, visual_dirs)
        }

        audio_dirs = dl_manager.download_and_extract(
            [_URLS["audio"].format(channel=channel) for channel in config_names]
        )
        audio_dict = {
            channel: audio_dir
            for channel, audio_dir in zip(config_names, audio_dirs)
        }

        transcript_dirs = dl_manager.download_and_extract(
            [_URLS["transcript"].format(channel=channel) for channel in config_names]
        )
        transcript_dict = {
            channel: transcript_dir
            for channel, transcript_dir in zip(config_names, transcript_dirs)
        }

        return [
            datasets.SplitGenerator(
                name=name,
                gen_kwargs={
                    "split": split,
                    "visual_dict": visual_dict,
                    "audio_dict": audio_dict,
                    "transcript_dict": transcript_dict,
                },
            )
            for name, split in split_dict.items()
        ]

    def _generate_examples(
        self, split: datasets.Dataset,
        visual_dict: dict,
        audio_dict: dict,
        transcript_dict: dict,
    ) -> Tuple[int, dict]:
        """
        Generate examples.
        :param split:                   Split.
        :param visual_dict:             Paths to directory containing visual files.
        :param audio_dict:              Paths to directory containing audio files.
        :param transcript_dict:         Paths to directory containing transcripts.
        :return:                        Example.
        """
        for i, sample in enumerate(split):
            channel = sample["channel"]
            visual_path = os.path.join(
                visual_dict[channel], channel, sample["id"] + ".mp4"
            )
            audio_path = os.path.join(
                audio_dict[channel], channel, sample["id"] + ".wav"
            )
            transcript_path = os.path.join(
                transcript_dict[channel], channel, sample["id"] + ".txt"
            )

            yield i, {
                "id": sample["id"],
                "channel": channel,
                "visual": self.__get_binary_data(visual_path),
                "duration": sample["duration"],
                "fps": sample["fps"],
                "audio": self.__get_binary_data(audio_path),
                "sampling_rate": sample["sampling_rate"],
                "transcript": self.__get_text_data(transcript_path),
            }

    def __get_binary_data(self, path: str) -> bytes:
        """
        Get binary data from path.
        :param path:    Path to file.
        :return:        Binary data.
        """
        with open(path, "rb") as f:
            return f.read()

    def __get_text_data(self, path: str) -> str:
        """
        Get transcript from path.
        :param path:     Path to transcript.
        :return:         Transcript.
        """
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
