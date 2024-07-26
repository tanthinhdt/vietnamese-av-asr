import torch
import logging
import traceback
from pathlib import Path
from loguru import logger
from argparse import Namespace
from configs import ModelConfig
from dataclasses import dataclass, field
from simple_parsing import ArgumentParser
from tools import load_model, get_dummy_input
from utils import config_logger, upload_to_hf


logging.root.setLevel(logging.WARNING)


def get_args() -> Namespace:
    parser = ArgumentParser(description="Export model to ONNX format")
    parser.add_arguments(ModelConfig, "config")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/onnx",
        help="Path to output ONNX file",
    )
    parser.add_argument(
        "--upload-to-hf",
        action="store_true",
        help="Upload ONNX file to Hugging Face",
    )
    return parser.parse_args()


@dataclass
class ONNXConfig:
    f: str
    dynamic_axes: dict
    input_names: list
    output_names: list = field(default_factory=lambda: ["logits"])
    do_constant_folding: float = True
    opset_version: int = 14


@dataclass
class AVSPLLMONNXConfig(ONNXConfig):
    input_names: list = field(default_factory=lambda: ["source"])
    dynamic_axes: dict = field(
        default_factory=lambda: {
            "source": {
                "video": {
                    0: "batch_size",
                    1: "num_frames",
                    2: "num_channels",
                    3: "height",
                    4: "width",
                }
            }
        }
    )


@dataclass
class AVHubertONNXConfig(ONNXConfig):
    input_names: list = field(default_factory=lambda: ["pixel_values"])
    dynamic_axes: dict = field(
        default_factory=lambda: {
            "pixel_values": {
                0: "batch_size",
                1: "num_frames",
                2: "num_channels",
                3: "height",
                4: "width",
            }
        }
    )


def main(args: Namespace) -> None:
    model_config = args.model
    logger.info(model_config)

    output_name = model_config.pretrained.split("/")[-1]
    output_file = Path(args.output_dir) / f"{output_name}.onnx"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    _, processor, model = load_model(model_config)
    logger.info("Model loaded")

    if model_config.arch == "avsp_llm":
        config_class = AVSPLLMONNXConfig
    else:
        raise Exception(f"Model {model_config.arch} is not supported")
    config = config_class(f=str(output_file))
    logger.info("Config loaded")

    batch_size = 1
    torch.onnx.export(
        model=model,
        args=get_dummy_input(model_config.arch, processor, batch_size),
        **vars(config),
    )
    logger.info(f"Model exported to {config.f}")

    if args.upload_to_hf:
        upload_to_hf(
            src_path=output_file,
            dest_path=output_file.name,
            repo_id=model_config.pretrained,
            repo_type="model",
        )
        logger.info(f"Model uploaded to {model_config.pretrained}")


if __name__ == "__main__":
    args = get_args()
    config_logger()

    try:
        main(args=args)
    except Exception:
        logger.error(f"Exporting model to ONNX interrupted:\n{traceback.format_exc()}")
