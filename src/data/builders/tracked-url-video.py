import os

import datasets
from huggingface_hub import HfFileSystem
from typing import Tuple, List
from colorama import Fore

logger = datasets.logging.get_logger(__name__)
fs = HfFileSystem()

_CITATION = """
"""
_DESCRIPTION = """
    This dataset contain url of video to download.
"""
_HOMEPAGE = "https://github.com/tanthinhdt/vietnamese-av-asr"
_REPO_PATH = "datasets/GSU24AI03-SU24AI21/tracked-url-video"
_BRANCH = 'main'
_REPO_PATH_BRANCH = f"{_REPO_PATH}@{_BRANCH}"
_REPO_URL = f"https://huggingface.co/{_REPO_PATH}/resolve/{_BRANCH}"

_URLS = {
    "metadata": f"{_REPO_URL}" + "/metadata/{channel}.parquet",
}

_CONFIGS = ["all"]
if fs.isdir(f"{_REPO_PATH_BRANCH}/metadata/"):
    _CONFIGS.extend([
        os.path.basename(file)[:-8]
        for file in fs.listdir(f"{_REPO_PATH_BRANCH}/metadata/", detail=False)
        if file.endswith('.parquet')
    ])


class TrackedUrlVideoConfig(datasets.BuilderConfig):
    """Tracked video url configuration."""

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


class TrackedUrlVideo(datasets.GeneratorBasedBuilder):
    """Tracked video url dataset."""

    BUILDER_CONFIGS = [TrackedUrlVideoConfig(name) for name in _CONFIGS]
    DEFAULT_CONFIG_NAME = "all"

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features({
            "id": datasets.Value("string"),
            "channel": datasets.Value("string"),
            "video_id": datasets.Value("string"),
            "url": datasets.Value("string"),
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

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    'metadata_paths': metadata_paths,
                },
            ),
        ]

    def _generate_examples(
            self,
            metadata_paths: List[str],
    ) -> Tuple[int, dict]: # type: ignore        
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
            video_id = sample['video_id']
            url = f'https://www.youtube.com/watch?v={video_id}'

            yield i, {
                "id": video_id,
                "channel": sample['channel'],
                "video_id": video_id,
                "url": url,
            }