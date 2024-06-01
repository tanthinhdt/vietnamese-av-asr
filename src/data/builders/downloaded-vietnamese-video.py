import os

import datasets
from huggingface_hub import HfFileSystem
from typing import List, Tuple

logger = datasets.logging.get_logger(__name__)
fs = HfFileSystem()

_CITATION = """
"""
_DESCRIPTION = """
    This dataset contains raw vietnamese sources video.
"""
_HOMEPAGE = "https://github.com/tanthinhdt/vietnamese-av-asr"

_METADATA_REPO_PATH = "datasets/GSU24AI03-SU24AI21/downloaded-vietnamese-video"
_VIDEO_REPO_PATH = "datasets/GSU24AI03-SU24AI21/downloaded-vietnamese-video"

_BRANCH = 'main'
_REPO_PATH_BRANCH = f"{_METADATA_REPO_PATH}@{_BRANCH}"

_REPO_URL = "https://huggingface.co/{}/resolve/{}"
_URLS = {
    "metadata": _REPO_URL.format(_METADATA_REPO_PATH,_BRANCH) + "/metadata/{channel}.parquet",
    "video": _REPO_URL.format(_VIDEO_REPO_PATH,_BRANCH)  + "/video/{channel}.zip",
}

_CONFIGS = ["all"]
if fs.isdir(f"{_REPO_PATH_BRANCH}/metadata/"):
    _CONFIGS.extend([
        os.path.basename(file)[:-8]
        for file in fs.listdir(f"{_REPO_PATH_BRANCH}/metadata/", detail=False)
        if file.endswith('.parquet')
    ])


class DownloadedVietnameseVideoConfig(datasets.BuilderConfig):
    """Raw Video configuration."""

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

class DownloadedVietnameseVideo(datasets.GeneratorBasedBuilder):
    """Raw Video dataset."""

    BUILDER_CONFIGS = [DownloadedVietnameseVideoConfig(name) for name in _CONFIGS]
    DEFAULT_CONFIG_NAME = "all"

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features({
            "id": datasets.Value("string"),
            "channel": datasets.Value("string"),
            "video_id": datasets.Value("string"),
            "video_path": datasets.Value("string"),
            "duration": datasets.Value("float"),
            "video_fps": datasets.Value("int8"),
            "audio_fps": datasets.Value("int32"),
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

        video_dirs = dl_manager.download_and_extract(
            [_URLS["video"].format(channel=channel) for channel in config_names]
        )

        video_dict = {
            channel: video_dir 
            for channel, video_dir in zip(config_names, video_dirs)
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    'metadata_paths': metadata_paths,
                    'video_dict': video_dict,
                },
            ),
        ]

    def _generate_examples(
            self,
            metadata_paths: List[str],
            video_dict: dict,
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
            trust_remote_code=False,
        )
        for i, sample in enumerate(dataset):
            video_path = os.path.join(
                video_dict[sample['channel']], sample['channel'], sample['video_name'] + ".mp4"
            )

            yield i, {
                'id': sample['video_id'],
                'channel': sample['channel'],
                'video_id': sample['video_id'],
                'video_path': video_path,
                'duration': sample['duration'],
                'video_fps': sample['video_fps'],
                'audio_fps': sample['audio_fps'],
            }