import logging
from loguru import logger
from argparse import Namespace
from simple_parsing import ArgumentParser
from configs import TestConfig
from tests import (
    test_wav2vec2_large_vi_vlsp2020,
    test_wav2vec2_base_vietnamese_250h,
)
from utils import config_logger


logging.root.setLevel(logging.WARNING)


test_models_dict = {
    "wav2vec2-large-vi-vlsp2020": test_wav2vec2_large_vi_vlsp2020,
    "wav2vec2-base-vietnamese-250h": test_wav2vec2_base_vietnamese_250h,
}


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_arguments(TestConfig, dest="config")
    return parser.parse_args()


def main(args: Namespace) -> None:
    config = args.config
    logger.info(config)

    assert config.model in test_models_dict, \
        f"Model {config.model} not found in {list(test_models_dict.keys())}"
    test_models_dict[config.model](config)

    logger.info("Done")


if __name__ == "__main__":
    args = get_args()
    config_logger(log_file=args.config.log_file)
    main(args=args)
