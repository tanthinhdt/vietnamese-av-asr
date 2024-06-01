import os

import datasets
from huggingface_hub import HfFileSystem
from typing import Tuple, List

logger = datasets.logging.get_logger(__name__)
fs = HfFileSystem()

_CITATION = """
"""
_DESCRIPTION = """
    This dataset contains clip active speaker (cropped face).
"""
_HOMEPAGE = "https://github.com/tanthinhdt/vietnamese-av-asr"

_METADATA_REPO_PATH = "datasets/GSU24AI03-SU24AI21/detected-speaker-clip"
_VISUAL_REPO_PATH = "datasets/GSU24AI03-SU24AI21/detected-speaker-clip"

_BRANCH = 'main'
_REPO_PATH_BRANCH = f"{_METADATA_REPO_PATH}@{_BRANCH}"

_REPO_URL = "https://huggingface.co/{}/resolve/{}"
_URLS = {
    "metadata": _REPO_URL.format(_METADATA_REPO_PATH,_BRANCH) + "/metadata/{channel}.parquet",
    "visual": _REPO_URL.format(_VISUAL_REPO_PATH,_BRANCH) + "/visual/{channel}.zip",
}

_CONFIGS = ["all"]
if fs.isdir(f"{_REPO_PATH_BRANCH}/metadata/"):
    _CONFIGS.extend([
        os.path.basename(file)[:-8]
        for file in fs.ls(f"{_REPO_PATH_BRANCH}/metadata/", detail=False)
        if file.endswith('.parquet')
    ])


class DetectdSpeakerClipConfig(datasets.BuilderConfig):
    """Detected active speaker clip configuration."""

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


class DetectdSpeakerClip(datasets.GeneratorBasedBuilder):
    """Detected active speaker clip dataset."""

    BUILDER_CONFIGS = [DetectdSpeakerClipConfig(name) for name in _CONFIGS]
    DEFAULT_CONFIG_NAME = "all"

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features({
            "id": datasets.Value("string"),
            "channel": datasets.Value("string"),
            "visual_path": datasets.Value("string"),
            "chunk_visual_id": datasets.Value("string"),
            "chunk_audio_id": datasets.Value("string"),
            "visual_num_frames": datasets.Value("float64"),
            "audio_num_frames": datasets.Value("float64"),
            "visual_fps": datasets.Value("int64"),
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

        visual_dirs = dl_manager.download_and_extract(
            [_URLS["visual"].format(channel=channel) for channel in config_names]
        )

        visual_dict = {
            channel: visual_dir
            for channel, visual_dir in zip(config_names, visual_dirs)
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    'metadata_paths': metadata_paths,
                    'visual_dict': visual_dict
                },
            ),
        ]

    def _generate_examples(
            self,
            metadata_paths: List[str],
            visual_dict: dict
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
            visual_path = os.path.join(
                visual_dict[sample['channel']], sample['channel'], sample['chunk_visual_id'] + ".mp4"
            )

            yield i, {
                'id': sample['id'],
                'channel': sample['channel'],
                'visual_path': visual_path,
                'chunk_visual_id': sample['chunk_visual_id'],
                'chunk_audio_id': sample['chunk_audio_id'],
                'visual_num_frames': sample['visual_num_frames'],
                'audio_num_frames': sample['audio_num_frames'],
                'visual_fps': sample['visual_fps'],
                'audio_fps': sample['audio_fps']
            }