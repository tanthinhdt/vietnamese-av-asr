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
    This dataset contain videos of Vietnamese speakers.
"""
_HOMEPAGE = "https://github.com/tanthinhdt/vietnamese-av-asr"
_REPO_PATH = "datasets/phdkhanh2507/vietnamese-speaker-video"
_REPO_URL = f"https://huggingface.co/{_REPO_PATH}/resolve/main"
_URLS = {
    "meta": f"{_REPO_URL}/metadata/" + "{channel}.parquet",
    "video": f"{_REPO_URL}/video/" + "{channel}.zip",
}
_CONFIGS = ["all"]
if fs.exists(_REPO_PATH + "/metadata"):
    _CONFIGS.extend([
        os.path.basename(file_name)[:-8]
        for file_name in fs.listdir(_REPO_PATH + "/metadata", detail=False)
        if file_name.endswith(".parquet")
    ])


class VietnameseSpeakerVideoConfig(datasets.BuilderConfig):
    """Vietnamese Speaker Video configuration."""

    def __init__(self, name, **kwargs) -> None:
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


class VietnameseSpeakerVideo(datasets.GeneratorBasedBuilder):
    """Vietnamese Speaker Video dataset."""

    BUILDER_CONFIGS = [VietnameseSpeakerVideoConfig(name) for name in _CONFIGS]
    DEFAULT_CONFIG_NAME = "all"

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features({
            "id": datasets.Value("string"),
            "channel": datasets.Value("string"),
            "video": datasets.Value("string"),
            "fps": datasets.Value("int8"),
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
        video_dirs = dl_manager.download_and_extract(
            [_URLS["video"].format(channel=channel) for channel in config_names]
        )

        video_dict = {
            channel: video_dir for channel, video_dir in zip(config_names, video_dirs)
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "metadata_paths": metadata_paths,
                    "video_dict": video_dict,
                },
            ),
        ]

    def _generate_examples(
        self, metadata_paths: List[str],
        video_dict: dict,
    ) -> Tuple[int, dict]:
        """
        Generate examples from metadata.
        :param metadata_paths:      Paths to metadata.
        :param video_dict:          Paths to directory containing videos.
        :yield:                     Example.
        """
        dataset = datasets.load_dataset(
            "parquet",
            data_files=metadata_paths,
            split="train",
            use_auth_token=True,
        )
        for i, sample in enumerate(dataset):
            channel = sample["channel"]
            video_path = os.path.join(
                video_dict[channel], channel, sample["id"] + ".avi"
            )

            yield i, {
                "id": sample["id"],
                "channel": channel,
                "video": video_path,
                "fps": sample["fps"],
                "sampling_rate": sample["sampling_rate"],
            }
