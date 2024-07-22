import torch
import logging
from loguru import logger
from argparse import Namespace
from traceback import format_exc
from simple_parsing import ArgumentParser
from huggingface_hub import hf_hub_download
from configs import ModelConfig
from tools import load_model
from utils import config_logger


logging.root.setLevel(logging.WARNING)


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

    checkpoint_file = hf_hub_download(
        repo_id=config.pretrained,
        filename="checkpoints/checkpoint_best.pt",
        repo_type="model",
        cache_dir="models/huggingface",
    )
    logger.info("Checkpoint file downloaded")

    model_state_dict = model.state_dict()
    model_state_dict.update(torch.load(checkpoint_file)["model"])
    model.load_state_dict(model_state_dict)
    logger.info("Model state dict loaded")

    model.push_to_hub(config.pretrained)
    logger.info("Model pushed to HuggingFace")


if __name__ == "__main__":
    args = get_args()
    config_logger()

    try:
        main(args=args)
    except Exception:
        logger.error(f"Registering model interrupted:\n{format_exc()}")
