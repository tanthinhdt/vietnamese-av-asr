import logging
import traceback
from pathlib import Path
from loguru import logger
from argparse import Namespace
from configs import ModelConfig
from simple_parsing import ArgumentParser
from utils import config_logger
from optimum.onnxruntime import ORTModelForCausalLM


logging.root.setLevel(logging.WARNING)


def get_args() -> Namespace:
    parser = ArgumentParser(description="Export model to ONNX format")
    parser.add_arguments(ModelConfig, "model")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/onnx",
        help="The repository ID to upload the model to.",
    )
    return parser.parse_args()


def main(args: Namespace) -> None:
    model_config = args.model
    logger.info(model_config)

    model = ORTModelForCausalLM.from_pretrained(
        model_config.pretrained,
        cache_dir="models/huggingface",
        trust_remote_code=True,
        export=True,
    )
    logger.info("Model loaded")

    output_dir = Path(args.output_dir) / model_config.pretrained.split("/")[-1]
    model.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")

    model.push_to_hub(output_dir, model_config.pretrained, private=True)
    logger.info(f"Model uploaded to {model_config.pretrained}")


if __name__ == "__main__":
    args = get_args()
    config_logger()

    try:
        main(args=args)
    except Exception:
        logger.error(f"Exporting LLM to ONNX interrupted:\n{traceback.format_exc()}")
