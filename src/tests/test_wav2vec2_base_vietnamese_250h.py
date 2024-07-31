import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from utils import compute_cer, compute_wer, compute_flops
from loguru import logger
from configs import TestConfig
from .utils import (
    load_test_set,
    log_params,
    log_average_results,
    save_results,
)


def get_input_values(
    processor: Wav2Vec2Processor,
    data: torch.Tensor,
    sampling_rate: int,
    device: str,
) -> torch.Tensor:
    input_values = processor(
        data,
        sampling_rate=sampling_rate,
        return_tensors="pt",
    )
    return input_values.input_values.to(device)


def evaluate(
    sample: dict,
    model: Wav2Vec2ForCTC,
    processor: Wav2Vec2Processor,
    device: str,
) -> dict:
    input_values = get_input_values(
        processor,
        data=sample["audio"]["array"],
        sampling_rate=sample["audio"]["sampling_rate"],
        device=device,
    )
    logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    sample["hyp"] = processor.batch_decode(predicted_ids)[0]

    sample["wer"] = compute_wer(sample["hyp"], sample["ref"])
    sample["cer"] = compute_cer(sample["hyp"], sample["ref"])

    return sample


def test_wav2vec2_base_vietnamese_250h(config: TestConfig) -> None:
    test_set = load_test_set(config.test_manifest_file, config.test_ref_file)

    repo_id = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
    logger.info(f"Model {repo_id}")

    processor = Wav2Vec2Processor.from_pretrained(
        repo_id,
        cache_dir=config.cache_dir,
    )
    logger.info("Processor loaded")

    model = Wav2Vec2ForCTC.from_pretrained(repo_id, cache_dir=config.cache_dir)
    model = model.eval().to(config.device)
    logger.info("Model loaded")
    log_params(model)

    sample = test_set[0]["audio"]
    input_values = get_input_values(
        processor=processor,
        data=sample["array"],
        sampling_rate=sample["sampling_rate"],
        device=config.device,
    )
    flops = compute_flops(model, inputs=(input_values,))
    logger.info(f"FLOPs: {flops:,}")
    logger.info(f"FLOPs (G): {flops / 1_000_000_000:.2f}")

    results = test_set.map(
        evaluate,
        fn_kwargs={
            "model": model,
            "processor": processor,
            "device": config.device,
        },
    )
    results = results.to_pandas()
    logger.info("Evaluation done")
    log_average_results(results)

    results_file = config.output_dir / "results.json"
    save_results(results, results_file)
