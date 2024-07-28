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


logging.root.setLevel(logging.WARNING)


def get_args() -> Namespace:
    parser = ArgumentParser(
        description="Register a model to HuggingFace.",
    )
    parser.add_arguments(ModelConfig, "model")
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="The repository ID to upload the model to.",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Whether to register the LLM.",
    )
    return parser.parse_args()


def main(args: Namespace) -> None:
    model_config = args.model
    logger.info(model_config)

    processor, model = load_model(model_config, do_train=True)
    logger.info("Model loaded")

    if Path(model_config.pretrained).exists():
        checkpoint_file = model_config.pretrained
        if args.repo_id is None:
            raise ValueError("Repository ID is required")
        repo_id = args.repo_id
    else:
        repo_id = args.repo_id if args.repo_id is not None else model_config.pretrained
        checkpoint_file = hf_hub_download(
            repo_id=model_config.pretrained,
            filename="checkpoints/checkpoint_best.pt",
            repo_type="model",
            cache_dir="models/huggingface",
        )
        logger.info("Checkpoint file downloaded")

    model_state_dict = model.state_dict()
    model_state_dict.update(torch.load(checkpoint_file)["model"])
    model.load_state_dict(model_state_dict)
    logger.info("Model state dict loaded")

    kmeans_model_file = Path(model.km_path)
    if kmeans_model_file.exists():
        api = HfApi()
        api.upload_file(
            path_or_fileobj=kmeans_model_file,
            path_in_repo=kmeans_model_file.name,
            repo_id=repo_id,
            repo_type="model",
        )
        logger.info("K-means model of AVSP-LLM uploaded")

    processor.push_to_hub(repo_id)
    model.push_to_hub(repo_id)
    logger.info(f"Model pushed to HuggingFace at {repo_id}")

    if args.llm:
        llm_repo_id = repo_id + "_LLM"
        model.decoder.push_to_hub(llm_repo_id)
        logger.info(f"LLM pushed to HuggingFace at {llm_repo_id}")


if __name__ == "__main__":
    args = get_args()
    config_logger()

    try:
        main(args=args)
    except Exception:
        logger.error(f"Registering model interrupted:\n{format_exc()}")
