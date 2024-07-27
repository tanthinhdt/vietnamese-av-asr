import torch
import logging
from pathlib import Path
from loguru import logger
from argparse import Namespace
from traceback import format_exc
from simple_parsing import ArgumentParser
from huggingface_hub import hf_hub_download, HfApi
from configs import ModelConfig
from tools import load_model
from utils import config_logger
from transformers import AutoModel


logging.root.setLevel(logging.WARNING)


def get_args() -> Namespace:
    parser = ArgumentParser(
        description="Register a model to HuggingFace.",
    )
    parser.add_arguments(ModelConfig, "config")
    return parser.parse_args()


def register_avsp_llm(
    config: ModelConfig,
    model: AutoModel,
) -> AutoModel:
    checkpoint_file = hf_hub_download(
        repo_id=config.pretrained,
        filename="checkpoints/checkpoint_best.pt",
        repo_type="model",
        cache_dir="models/huggingface",
    )
    logger.info("AVSP-LLM checkpoint file downloaded")

    model_state_dict = model.state_dict()
    model_state_dict.update(torch.load(checkpoint_file)["model"])
    model.load_state_dict(model_state_dict)
    logger.info("AVSP-LLM model state dict loaded")

    kmeans_model_file = Path(model.km_path)
    if kmeans_model_file.exists():
        api = HfApi()
        api.upload_file(
            path_or_fileobj=kmeans_model_file,
            path_in_repo=kmeans_model_file.name,
            repo_id=config.pretrained,
            repo_type="model",
        )
        logger.info("K-means model of AVSP-LLM uploaded")

    return model


def main(args: Namespace) -> None:
    config = args.config
    logger.info(config)

    _, processor, model = load_model(config, do_train=True)
    logger.info("Model loaded")

    if config.arch == "avsp_llm":
        model = register_avsp_llm(config, model)
    else:
        raise NotImplementedError(f"Model {config.arch} is not supported")

    processor.push_to_hub(config.pretrained)
    model.push_to_hub(config.pretrained)
    logger.info("Model pushed to HuggingFace")


if __name__ == "__main__":
    args = get_args()
    config_logger()

    try:
        main(args=args)
    except Exception:
        logger.error(f"Registering model interrupted:\n{format_exc()}")
