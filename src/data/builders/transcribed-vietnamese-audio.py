import os
import datasets
from huggingface_hub import HfFileSystem
from typing import List, Tuple


logger = datasets.logging.get_logger(__name__)
fs = HfFileSystem()


_CITATION = """

"""
_DESCRIPTION = """
    This dataset contains transcripts from audio of Vietnamese speakers.
"""
_HOMEPAGE = "https://github.com/tanthinhdt/vietnamese-av-asr"
_MAIN_REPO_PATH = "datasets/GSU24AI03-SU24AI21/transcribed-vietnamese-audio"
_VISUAL_REPO_PATH = "datasets/GSU24AI03-SU24AI21/detected-vietnamese-clip"
_REPO_URL = "https://huggingface.co/{}/resolve/main"
_URLS = {
    "meta": f"{_REPO_URL}/metadata/".format(_MAIN_REPO_PATH) + "{channel}.parquet",
    "visual": f"{_REPO_URL}/visual/".format(_VISUAL_REPO_PATH) + "{channel}.zip",
}
_CONFIGS = ["all"]
if fs.exists(_MAIN_REPO_PATH + "/metadata"):
    _CONFIGS.extend([
        os.path.basename(file_name)[:-8]
        for file_name in fs.listdir(_MAIN_REPO_PATH + "/metadata", detail=False)
        if file_name.endswith(".parquet")
    ])


class TranscribedVietnameseAudioConfig(datasets.BuilderConfig):
    """Transcribed Vietnamese Audio configuration."""

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


class TranscribedVietnameseAudio(datasets.GeneratorBasedBuilder):
    """Transcribed Vietnamese Audio dataset."""

    BUILDER_CONFIGS = [TranscribedVietnameseAudioConfig(name) for name in _CONFIGS]
    DEFAULT_CONFIG_NAME = "all"

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features({
            "id": datasets.Value("string"),
            "channel": datasets.Value("string"),
            "chunk_visual_id": datasets.Value("string"),
            "chunk_audio_id": datasets.Value("string"),
            "visual": datasets.Value("string"),
            "visual_num_frames": datasets.Value("float64"),
            "audio_num_frames": datasets.Value("float64"),
            "visual_fps": datasets.Value("int64"),
            "audio_fps": datasets.Value("int64"),
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
        visual_dirs = dl_manager.download_and_extract(
            [_URLS["visual"].format(channel=channel) for channel in config_names]
        )

        visual_dict = {
            channel: visual_dir for channel, visual_dir in zip(config_names, visual_dirs)
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "metadata_paths": metadata_paths,
                    "visual_dict": visual_dict,
                },
            ),
        ]

    def _generate_examples(
        self, metadata_paths: List[str],
        visual_dict: dict,
    ) -> Tuple[int, dict]:
        """
        Generate examples from metadata.
        :param metadata_paths:      Paths to metadata.
        :param audio_dict:          Paths to directory containing audios.
        :param transcript_dict:     Paths to directory containing transcripts.
        :yield:                     Example.
        """
        dataset = datasets.load_dataset(
            "parquet",
            data_files=metadata_paths,
            split="train",
        )
        for i, sample in enumerate(dataset):
            channel = sample["channel"]
            visual_path = os.path.join(
                visual_dict[channel], channel, sample["chunk_visual_id"] + ".mp4"
            )

            yield i, {
                "id": sample["id"],
                "channel": channel,
                "visual": visual_path,
                "chunk_visual_id": sample["chunk_visual_id"],
                "chunk_audio_id": sample["chunk_audio_id"],
                "visual_num_frames": sample["visual_num_frames"],
                "audio_num_frames": sample["audio_num_frames"],
                "visual_fps": sample["visual_fps"],
                "audio_fps": sample["audio_fps"],
                "transcript": sample["transcript"],
            }