# Copyright 2023 Thinh T. Duong
import os
import datasets
from huggingface_hub import HfFileSystem


logger = datasets.logging.get_logger(__name__)
fs = HfFileSystem()


_CITATION = """

"""
_DESCRIPTION = """
    This dataset contain videos of Vietnamese speakers.
"""
_HOMEPAGE = "https://github.com/duytran1332002/vlr"
_REPO_PATH = "datasets/fptu/vietnamese-video"
_REPO_URL = f"https://huggingface.co/{_REPO_PATH}/resolve/main"
_URLS = {
    "meta": f"{_REPO_URL}/metadata/" + "{channel}.parquet",
}
_CONFIGS = ["all"]
if fs.exists(_REPO_PATH + "/metadata"):
    _CONFIGS.extend([
        os.path.basename(file_name)[:-8]
        for file_name in fs.listdir(_REPO_PATH + "/metadata", detail=False)
        if file_name.endswith(".parquet")
    ])


class VietnameseVideoConfig(datasets.BuilderConfig):
    """Vietnamese Speaker Video configuration."""

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


class VietnameseVideo(datasets.GeneratorBasedBuilder):
    """Vietnamese Speaker Video dataset."""

    BUILDER_CONFIGS = [VietnameseVideoConfig(name) for name in _CONFIGS]
    DEFAULT_CONFIG_NAME = "all"

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features({
            "id": datasets.Value("string"),
            "channel": datasets.Value("string"),
            "video": datasets.Value("string"),
        })

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> list[datasets.SplitGenerator]:
        """
        Get splits.
        :param dl_manager:  Download manager.
        :return:            Splits.
        """
        config_names = _CONFIGS[1:] if self.config.name == "all" else [self.config.name]

        metadata_paths = dl_manager.download(
            [_URLS["meta"].format(channel=channel) for channel in config_names]
        )

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "metadata_paths": metadata_paths,
                },
            ),
        ]

    def _generate_examples(
        self, metadata_paths: list[str],
    ) -> tuple[int, dict]:
        """
        Generate examples from metadata.
        :param metadata_paths:      Paths to metadata.
        :yield:                     Example.
        """
        dataset = datasets.load_dataset(
            "parquet",
            data_files=metadata_paths,
            split="train",
        )
        for i, sample in enumerate(dataset):
            yield i, sample
