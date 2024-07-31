import torch
import pandas as pd
from pathlib import Path
from datasets import Dataset, Audio
from loguru import logger


def load_test_set(test_manifest_file: str, test_ref_file: str) -> Dataset:
    with open(test_manifest_file) as f:
        root_dir = Path(f.readline().strip())
        audio_files = [str(root_dir / line.split()[2]) for line in f.readlines()]
    with open(test_ref_file) as f:
        refs = [line.strip() for line in f.readlines()]
    test_set = Dataset.from_dict({"audio": audio_files, "ref": refs})
    return test_set.cast_column("audio", Audio())


def log_params(model: torch.nn.Module) -> None:
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Num trainable params: {num_params:,}")

    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Num params: {num_trainable_params:,}")


def log_average_results(results: pd.DataFrame) -> None:
    avg_wer = results["wer"].mean()
    avg_cer = results["cer"].mean()
    logger.info(f"Average WER: {avg_wer:.2f}")
    logger.info(f"Average CER: {avg_cer:.2f}")


def save_results(results: pd.DataFrame, results_file: Path) -> None:
    reduced_results = results[["ref", "hyp", "cer", "wer"]]
    reduced_results.to_json(results_file, default_handler=str)
    logger.info(f"Results saved to {results_file}")
