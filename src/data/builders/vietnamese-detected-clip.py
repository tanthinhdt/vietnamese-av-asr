import os

import datasets
from huggingface_hub import HfFileSystem
from typing import Tuple, List

logger = datasets.logging.get_logger(__name__)
fs = HfFileSystem()

_CITATION = """
"""
_DESCRIPTION = """
    This dataset contain raw video of Vietnamese speakers.
"""
_HOMEPAGE = "https://github.com/tanthinhdt/vietnamese-av-asr"
_REPO_PATH = "datasets/GSU24AI03-SU24AI21/vietnamese-detected-clip"
_BRANCH = 'main'
_REPO_PATH_BRANCH = f"{_REPO_PATH}@{_BRANCH}"
_REPO_URL = f"https://huggingface.co/{_REPO_PATH}/resolve/{_BRANCH}"

_URLS = {
    "video": f"{_REPO_URL}" + "/video/{channel}.zip",
    "audio": f"{_REPO_URL}" + "/audio/{channel}.zip",
    "metadata": f"{_REPO_URL}" + "/metadata/{channel}.parquet",
}

_CONFIGS = ["all"]
_CONFIGS.extend([
    os.path.basename(file)[:-8]
    for file in fs.ls(f"{_REPO_PATH_BRANCH}/metadata/", detail=False)
    if file.endswith('.parquet')
])


class VietnameseDetectedClipConfig(datasets.BuilderConfig):
    """Raw Vietnamese Clip configuration."""

    def __init__(self, name: str, **kwargs):
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


class VietnameseDetectedClip(datasets.GeneratorBasedBuilder):
    """Raw Vietnamese Clip dataset."""

    BUILDER_CONFIGS = [VietnameseDetectedClipConfig(name) for name in _CONFIGS]
    DEFAULT_CONFIG_NAME = "all"

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features({
            "id": datasets.Value("string"),
            "channel": datasets.Value("string"),
            "video_id": datasets.Value("string"),
            "video_path": datasets.Value("string"),
            "audio_path": datasets.Value("string"),
            "video_num_frames": datasets.Value("float64"),
            "audio_num_frames": datasets.Value("float64"),
            "video_fps": datasets.Value("int64"),
            "audio_fps": datasets.Value("int64"),
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
        config_names: List[str] = _CONFIGS[1:] if self.config.name == 'all' else [self.config.name]

        metadata_paths = dl_manager.download(
            [_URLS["metadata"].format(channel=channel) for channel in config_names]
        )

        video_paths = dl_manager.download_and_extract(
            [_URLS["video"].format(channel=channel) for channel in config_names]
        )

        audio_paths = dl_manager.download_and_extract(
            [_URLS["audio"].format(channel=channel) for channel in config_names]
        )

        data_dict = {
            channel: (video_paths, audio_path)
            for channel, video_path, audio_path in zip(config_names, video_paths, audio_paths)
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    'metadata_paths': metadata_paths,
                    'data_dict': data_dict
                },
            ),
        ]

    def _generate_examples(
            self,
            metadata_paths: List[str],
            data_dict: dict
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
            video_path = os.path.join(
                data_dict[sample['channel']][0], sample['channel'], sample['video_path']
            )

            audio_path = os.path.join(
                data_dict[sample['channel']][1], sample['channel'], sample['video_path']
            )

            yield i, {
                'id': sample['id'],
                'channel': sample['channel'],
                'video_id': sample['video_id'],
                'video_path': video_path,
                'audio_path': audio_path,
                'video_num_frames': sample['video_num_frames'],
                'audio_num_frames': sample['audio_num_frames'],
                'video_fps': sample['video_fps'],
                'audio_fps': sample['audio_fps']
            }

