import os

import datasets
from huggingface_hub import HfFileSystem
from typing import List, Tuple

import pickle

logger = datasets.logging.get_logger(__name__)
fs = HfFileSystem()

_CITATION = """
"""
_DESCRIPTION = """
    This dataset contain raw video of Vietnamese speakers.
"""
_HOMEPAGE = "https://github.com/tanthinhdt/vietnamese-av-asr"
_REPO_PATH = "datasets/GSU24AI03-SU24AI21/downloaded-vietnamese-video"
_BRANCH = 'raw_data'
_REPO_PATH_BRANCH = f"{_REPO_PATH}@{_BRANCH}"
_REPO_URL = f"https://huggingface.co/{_REPO_PATH}/resolve/{_BRANCH}"
_REPO_CONFIGS_URL = f"https://huggingface.co/{_REPO_PATH}/resolve/main/split_id.pckl"

_PATHS = {
    "video": f"{_REPO_PATH_BRANCH}" + "/raw/*/{video_id}/video.mp4",
    "metadata": f"{_REPO_PATH_BRANCH}" + "/raw/*/{video_id}/video.info.json",
}


with open(_REPO_CONFIGS_URL, 'rb') as f:
    SPLIT_ID: dict = pickle.load(f)

_CONFIGS = ["all"]
_CONFIGS.extend(list(SPLIT_ID.keys()))


class RawVietnameseClipConfig(datasets.BuilderConfig):
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


class RawVietnameseClip(datasets.GeneratorBasedBuilder):
    """Raw Vietnamese Clip dataset."""

    BUILDER_CONFIGS = [RawVietnameseClipConfig(name) for name in _CONFIGS]
    DEFAULT_CONFIG_NAME = "all"

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features({
            "id": datasets.Value("string"),
            "channel": datasets.Value("string"),
            "video_id": datasets.Value("string"),
            "video_path": datasets.Value("string"),
            "duration": datasets.Value("float16"),
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

        data_dict = dict()
        for channel in config_names:
            metadata_paths = []
            for video_id in SPLIT_ID[channel]:
                metadata_repo_path = fs.glob(_PATHS['metadata'].format(video_id=video_id), detail=False)[0]
                metadata_url = metadata_repo_path.replace(_REPO_PATH_BRANCH, _REPO_URL)
                metadata_paths.extend(dl_manager.download([metadata_url]))

            data_dict[channel] = metadata_paths

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    'data_dict': data_dict,
                    'dl_manager': dl_manager
                },
            ),
        ]

    def _generate_examples(
            self,
            data_dict: dict,
            dl_manager: datasets.DownloadManager,
    ) -> Tuple[int, dict]:
        """
        Generate examples from metadata.
        :param metadata_paths:      Paths to metadata.
        :param audio_dict:          Paths to directory containing audio.
        :yield:                     Example.
        """
        for channel in data_dict:
            dataset = datasets.load_dataset(
                "json",
                data_files=data_dict[channel],
                split="train",
                trust_remote_code=False,
            )
            for i, sample in enumerate(dataset):
                video_id = sample['video_id']
                video_repo_path = fs.glob(_PATHS['video'].format(video_id=video_id), detail=False)[0]

                video_repo_url = video_repo_path.replace(_REPO_PATH_BRANCH, _REPO_URL)
                video_local_path_tmp = dl_manager.download(video_repo_url)
                cache_dir, hash_id = os.path.split(video_local_path_tmp)
                os.remove(video_local_path_tmp)
                os.remove(video_local_path_tmp + '.json')
                os.remove(video_local_path_tmp + '.lock')

                local_path = os.path.join(cache_dir, channel, '')
                os.makedirs(local_path, exist_ok=True)
                local_path = os.path.join(local_path, f"{channel}@{video_id}.mp4")
                fs.get(rpath=video_repo_path, lpath=local_path, recursive=True)

                yield i, {
                    'id': hash_id,
                    'channel': channel,
                    'video_id': sample['video_id'],
                    'video_path': local_path,
                    'duration': sample['duration'],
                    'video_num_frames': sample['video_num_frames'],
                    'audio_num_frames': sample['audio_num_frames'],
                    'video_fps': sample['video_fps'],
                    'audio_fps': sample['audio_fps']
                }