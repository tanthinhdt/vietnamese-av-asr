import os
import datasets
from huggingface_hub import HfFileSystem
from typing import List, Tuple

logger = datasets.logging.get_logger(__name__)
fs = HfFileSystem()

_CITATION = """
"""
_DESCRIPTION = """
    This dataset contains the mouth region of speakers.
"""
_HOMEPAGE = "https://github.com/tanthinhdt/vietnamese-av-asr"

_METADATA_REPO_PATH = "datasets/GSU24AI03-SU24AI21/cropped-mouth-clip"
_AUDIO_REPO_PATH = "datasets/GSU24AI03-SU24AI21/detected-speaker-clip"

_BRANCH = 'main'
_REPO_BRANCH_PATH = f"{_METADATA_REPO_PATH}@{_BRANCH}"

_REPO_URL = "https://huggingface.co/{}/resolve/{}"
_URLS = {
    "metadata": _REPO_URL.format(_METADATA_REPO_PATH,_BRANCH) + "/metadata/{channel}.parquet",
    "audio": _REPO_URL.format(_AUDIO_REPO_PATH,_BRANCH) + "/audio/{channel}.zip",
}

_CONFIGS = ["all"]
if fs.exists(_REPO_BRANCH_PATH + "/metadata"):
    _CONFIGS.extend([
        os.path.basename(file_name)[:-8]
        for file_name in fs.listdir(_REPO_BRANCH_PATH + "/metadata", detail=False)
        if file_name.endswith(".parquet")
    ])


class CroppedMouthClipConfig(datasets.BuilderConfig):
    """Cropped mouth of speaker clip configuration."""

    def __init__(self, name, **kwargs):
        """
        Config for subset.

        name:
            Name of subset.
        kwargs:
            Arguments.
        """
        super(CroppedMouthClipConfig, self).__init__(
            name=name,
            version=datasets.Version("1.0.0"),
            description=_DESCRIPTION,
            **kwargs,
        )


class CroppedMouthClip(datasets.GeneratorBasedBuilder):
    """Cropped mouth of speaker clip dataset."""

    BUILDER_CONFIGS = [CroppedMouthClipConfig(name) for name in _CONFIGS]
    DEFAULT_CONFIG_NAME = "all"

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features({
            "id":                   datasets.Value("string"),
            "channel":              datasets.Value("string"),
            "audio_path":           datasets.Value("string"),
            "chunk_visual_id":      datasets.Value("string"),
            "chunk_audio_id":       datasets.Value("string"),
            "visual_num_frames":    datasets.Value("float64"),
            "audio_num_frames":     datasets.Value("float64"),
            "visual_fps":           datasets.Value("int64"),
            "audio_fps":            datasets.Value("int64"),
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

        dl_manager:
            Download manager.
        return:
            Splits.
        """
        config_names = _CONFIGS[1:] if self.config.name == "all" else [self.config.name]

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
                    "metadata_paths": metadata_paths,
                    "audio_dict": audio_dict,
                },
            ),
        ]

    def _generate_examples(
        self, metadata_paths: List[str],
        audio_dict: dict,
    ) -> Tuple[int, dict]: # type: ignore
        """ 
        Generate examples from metadata.

        metadata_paths:    
            Paths to metadata.
        audio_dict:       
            Paths to directory containing videos.
        yield:
            Example.
        """
        dataset = datasets.load_dataset(
            "parquet",
            data_files=metadata_paths,
            split="train",
        )
        for i, sample in enumerate(dataset):
            audio_path = os.path.join(
                audio_dict[sample["channel"]], sample["channel"], sample["chunk_audio_id"] + ".wav"
            )

            yield i, {
                "id":                   sample["id"],
                "channel":              sample["channel"],
                "audio_path":           audio_path,
                "chunk_visual_id":      sample["chunk_visual_id"],
                "chunk_audio_id":       sample["chunk_audio_id"],
                "visual_num_frames":    sample["visual_num_frames"],
                "audio_num_frames":     sample["audio_num_frames"],
                "visual_fps":           sample["visual_fps"],
                "audio_fps":            sample["audio_fps"],
            }