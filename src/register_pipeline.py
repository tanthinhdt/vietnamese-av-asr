import logging
import traceback
from loguru import logger
from tools import load_pipeline
from configs import ModelConfig
from simple_parsing import ArgumentParser
from argparse import Namespace
from utils import config_logger


logging.root.setLevel(logging.WARNING)


def get_args() -> Namespace:
    parser = ArgumentParser(
        description="Register a pipeline to HuggingFace.",
    )
    parser.add_arguments(ModelConfig, "config")
    return parser.parse_args()


def main(args: Namespace) -> None:
    model_config = args.config
    logger.info(model_config)

    pipeline = load_pipeline(model_config)
    logger.info("Pipeline loaded")

    pipeline.push_to_hub(model_config.pretrained)
    logger.info("Pipeline pushed to HuggingFace")


if __name__ == "__main__":
    args = get_args()
    config_logger()

    try:
        main(args=args)
    except Exception:
        logger.info(f"Registering pipeline interrupted:\n{traceback.format_exc()}")
