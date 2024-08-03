import logging
from argparse import Namespace
from dataclasses import dataclass
from simple_parsing import ArgumentParser, subgroups
from data import (
    create_manifest,
    dump_feature,
    learn_kmeans,
    dump_label,
    count_clusters,
)
from loguru import logger
from visualization import save_frames
from configs import (
    ProcessConfig,
    SaveFramesConfig,
    CreateManifestConfig,
    DumpFeatureConfig,
    LearnKMeansConfig,
    DumpLabelConfig,
    CountClustersConfig,
)


logging.root.setLevel(logging.WARNING)


@dataclass
class Config:
    process: ProcessConfig = subgroups(
        {
            "save_frames": SaveFramesConfig,
            "create_manifest": CreateManifestConfig,
            "dump_feature": DumpFeatureConfig,
            "learn_kmeans": LearnKMeansConfig,
            "dump_label": DumpLabelConfig,
            "count_clusters": CountClustersConfig,
        },
        default="create_manifest",
    )


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_arguments(Config, dest="config")
    args = parser.parse_args()
    return args


def main(args: Namespace) -> None:
    config = args.config.process
    logger.info(config)

    if isinstance(config, CreateManifestConfig):
        create_manifest(config)
    elif isinstance(config, DumpFeatureConfig):
        dump_feature(config)
    elif isinstance(config, LearnKMeansConfig):
        learn_kmeans(config)
    elif isinstance(config, DumpLabelConfig):
        dump_label(config)
    elif isinstance(config, CountClustersConfig):
        count_clusters(config)
    elif isinstance(config, SaveFramesConfig):
        save_frames(config)


if __name__ == "__main__":
    args = get_args()
    main(args=args)
