import torch
import logging
from loguru import logger
from argparse import Namespace
from simple_parsing import ArgumentParser
from configs import ModelConfig
from tempfile import TemporaryDirectory
from tools import load_model
from utils import config_logger, download_from_hf


config_logger()


def get_args() -> Namespace:
    parser = ArgumentParser(
        description="Register a model to HuggingFace.",
    )
    parser.add_arguments(ModelConfig, "config")
    return parser.parse_args()


def main(args: Namespace) -> None:
    config = args.config
    logger.info(config)

    _, _, model = load_model(config, do_train=True)
    logger.info("Model loaded")

    with TemporaryDirectory() as tmp_dir:
        checkpoint_file = download_from_hf(
            repo_id=config.pretrained,
            path_in_repo="checkpoints/checkpoint_best.pt",
            output_dir=tmp_dir,
            repo_type="model",
        )

        model_state_dict = torch.load(checkpoint_file)["model"]
        model_state_dict = model.state_dict().update(model_state_dict)
        model.load_state_dict(model_state_dict)
        logging.info("Model state dict loaded")

        model.push_to_hub(config.pretrained)
        logging.info("Model pushed to HuggingFace")


if __name__ == "__main__":
    args = get_args()
    main(args=args)
