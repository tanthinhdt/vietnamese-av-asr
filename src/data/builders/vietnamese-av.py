import os
import datasets
from huggingface_hub import HfFileSystem
from typing import List, Tuple

logger = datasets.logging.get_logger(__name__)
fs = HfFileSystem()

_CITATION = """
"""
_DESCRIPTION = """
        This dataset contains Vietnamese speakers clip along transcripts.
"""
_HOMEPAGE = "https://github.com/tanthinhdt/vietnamese-av-asr"

_METADATA_REPO_PATH = "datasets/GSU24AI03-SU24AI21/transcribed-vietnamese-audio"
_VISUAL_REPO_PATH = "datasets/GSU24AI03-SU24AI21/cropped-mouth-clip"
_AUDIO_REPO_PATH = "datasets/GSU24AI03-SU24AI21/detected-vietnamese-clip"

_BRANCH = "main"
_REPO_BRANCH_PATH = f"{_METADATA_REPO_PATH}@{_BRANCH}"

_REPO_URL = "https://huggingface.co/{}/resolve/{}"
_URLS = {
    "metadata": _REPO_URL.format(_METADATA_REPO_PATH,_BRANCH) + "/metadata/{channel}.parquet",
    "visual": _REPO_URL.format(_VISUAL_REPO_PATH,_BRANCH) + "/visual/{channel}.zip",
    "audio": _REPO_URL.format(_AUDIO_REPO_PATH,_BRANCH) + "/audio/{channel}.zip",
}

_CONFIGS = ["all"]
if fs.exists(_REPO_BRANCH_PATH + "/metadata"):
    _CONFIGS.extend([
        os.path.basename(file_name)[:-8]
        for file_name in fs.listdir(_REPO_BRANCH_PATH + "/metadata", detail=False)
        if file_name.endswith(".parquet")
    ])


class VietnameseAVConfig(datasets.BuilderConfig):
    """VietnameseAV configuration."""

    def __init__(self, name, **kwargs):
        """
        Config for subset.

        name:
            Name of subset.
        kwargs:
            Arguments.
        """
        super().__init__(
            name=name,
            version=datasets.Version("1.0.0"),
            description=_DESCRIPTION,
            **kwargs,
        )


class VietnameseAV(datasets.GeneratorBasedBuilder):
    """VietnameseAV dataset."""

    BUILDER_CONFIGS = [VietnameseAVConfig(name) for name in _CONFIGS]

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features({
            "id":                   datasets.Value("string"),
            "channel":              datasets.Value("string"),
            "visual":               datasets.Value("binary"),
            "audio":                datasets.Value("binary"),
            "chunk_visual_id":      datasets.Value("string"),
            "chunk_audio_id":       datasets.Value("string"),
            "visual_num_frames":    datasets.Value("float64"),
            "audio_num_frames":     datasets.Value("float64"),
            "visual_fps":           datasets.Value("int64"),
            "audio_fps":            datasets.Value("int64"),
            "transcript":           datasets.Value("string"),
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
        channel_demo = 'batch_88888'
        if channel_demo in _CONFIGS:
            _CONFIGS.remove(channel_demo)

        config_names = _CONFIGS[1:] if self.config.name == "all" else [self.config.name]

        metadata_paths = dl_manager.download(
            [_URLS["metadata"].format(channel=channel) for channel in config_names]
        )

        dataset = datasets.load_dataset(
            "parquet",
            data_files=metadata_paths,
            split="train",
        )

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

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "dataset": dataset,
                    "visual_dict": visual_dict,
                    "audio_dict": audio_dict,
                },
            )
        ]

    def _generate_examples(
        self, dataset: datasets.Dataset,
        visual_dict: dict,
        audio_dict: dict,
    ) -> Tuple[int, dict]:  # type: ignore
        """
        Generate examples.

        dataset:
            Dataset.
        visual_dict:
            Paths to directory containing visual files.
        audio_dict:
            Paths to directory containing audio files.
        yield:
            Example.
        """
        for i, sample in enumerate(dataset):
            channel = sample["channel"]
            visual_path = os.path.join(
                visual_dict[channel], channel, sample["chunk_visual_id"] + ".mp4"
            )
            audio_path = os.path.join(
                audio_dict[channel], channel, sample["chunk_audio_id"] + ".wav"
            )

            yield i, {
                "id":                   sample["id"],
                "channel":              channel,
                "visual":               self.__get_binary_data(visual_path),
                "audio":                self.__get_binary_data(audio_path),
                "chunk_visual_id":      sample["chunk_visual_id"],
                "chunk_audio_id":       sample["chunk_audio_id"],
                "visual_num_frames":    sample["visual_num_frames"],
                "audio_num_frames":     sample["audio_num_frames"],
                "visual_fps":           sample["visual_fps"],
                "audio_fps":            sample["audio_fps"],
                "transcript":           sample["transcript"],
            }

    def __get_binary_data(self, path: str) -> bytes:
        """
        Get binary data from path.

        path:
            Path to file.
        return:
            Binary data.
        """
        with open(path, "rb") as f:
            return f.read()
