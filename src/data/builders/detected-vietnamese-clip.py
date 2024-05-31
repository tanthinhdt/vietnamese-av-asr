import os

import datasets
from huggingface_hub import HfFileSystem
from typing import Tuple, List

logger = datasets.logging.get_logger(__name__)
fs = HfFileSystem()

_CITATION = """
"""
_DESCRIPTION = """
    This dataset contains Vietnamese speakers's cropped mouth clip.
"""
_HOMEPAGE = "https://github.com/tanthinhdt/vietnamese-av-asr"
_REPO_PATH = "datasets/GSU24AI03-SU24AI21/detected-vietnamese-clip"
_BRANCH = 'main'
_REPO_PATH_BRANCH = f"{_REPO_PATH}@{_BRANCH}"
_REPO_URL = f"https://huggingface.co/{_REPO_PATH}/resolve/{_BRANCH}"

_URLS = {
    "audio": f"{_REPO_URL}" + "/audio/{channel}.zip",
    "metadata": f"{_REPO_URL}" + "/metadata/{channel}.parquet",
}

_CONFIGS = ["all"]
if fs.exists(f"{_REPO_PATH_BRANCH}/metadata/"):
    _CONFIGS.extend([
        os.path.basename(file)[:-8]
        for file in fs.ls(f"{_REPO_PATH_BRANCH}/metadata/", detail=False)
        if file.endswith('.parquet')
    ])


class VietnameseDetectedClipConfig(datasets.BuilderConfig):
    """Vietnamese speakers's cropped mouth configuration."""

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
    """Vietnamese speakers's cropped mouth configuration dataset."""

    BUILDER_CONFIGS = [VietnameseDetectedClipConfig(name) for name in _CONFIGS]
    DEFAULT_CONFIG_NAME = "all"

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features({
            "id": datasets.Value("string"),
            "channel": datasets.Value("string"),
            "chunk_audio_id": datasets.Value("string"),
            "audio_path": datasets.Value("string"),
            "audio_num_frames": datasets.Value("float64"),
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

        audio_dirs = dl_manager.download_and_extract(
            [_URLS["audio"].format(channel=channel) for channel in config_names]
        )

        audio_dict = {
            channel: audio_dir
            for channel, audio_dir in zip(config_names, audio_dirs)
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    'metadata_paths': metadata_paths,
                    'audio_dict': audio_dict
                },
            ),
        ]

    def _generate_examples(
            self,
            metadata_paths: List[str],
            audio_dict: dict
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
            audio_path = os.path.join(
                audio_dict[sample['channel']], sample['channel'], sample['chunk_audio_id'] + ".wav"
            )

            yield i, {
                'id': sample['id'],
                'channel': sample['channel'],
                'chunk_audio_id': sample['chunk_audio_id'],
                'audio_path': audio_path,
                'audio_num_frames': sample['audio_num_frames'],
                'audio_fps': sample['audio_fps']
            }