import torch
from transformers import Wav2Vec2ProcessorWithLM, Wav2Vec2ForCTC
from huggingface_hub import hf_hub_download
from importlib.machinery import SourceFileLoader
from utils import compute_cer, compute_wer, compute_flops
from loguru import logger
from configs import TestConfig
from .utils import load_test_set, log_params


def get_input_values(
    processor: Wav2Vec2ProcessorWithLM,
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
    processor: Wav2Vec2ProcessorWithLM,
    device: str,
    with_lm: bool = False,
) -> dict:
    input_values = get_input_values(
        processor,
        sample["audio"]["array"],
        sampling_rate=sample["audio"]["sampling_rate"],
        device=device,
    )
    logits = model(input_values).logits

    if with_lm:
        sample["hyp"] = processor.decode(
            logits.cpu().detach().numpy()[0],
            beam_width=100,
        ).text
    else:
        sample["hyp"] = processor.tokenizer.decode(
            logits.argmax(dim=-1)[0].detach().cpu().numpy()
        )

    sample["wer"] = compute_wer(sample["hyp"], sample["ref"])
    sample["cer"] = compute_cer(sample["hyp"], sample["ref"])

    return sample


def test_wav2vec2_large_vi_vlsp2020(config: TestConfig) -> None:
    test_set = load_test_set(config.test_manifest_file, config.test_ref_file)
    logger.info("Test set loaded")
    logger.info(f"Num samples: {len(test_set)}")

    repo_id = "nguyenvulebinh/wav2vec2-large-vi-vlsp2020"
    logger.info(f"Model: {repo_id}")
    logger.info(f"With LM: {config.with_lm}")

    processor = Wav2Vec2ProcessorWithLM.from_pretrained(
        repo_id,
        cache_dir=config.cache_dir,
    )
    logger.info("Processor loaded")

    model = SourceFileLoader(
        "model",
        hf_hub_download(
            repo_id=repo_id,
            filename="model_handling.py",
            cache_dir=config.cache_dir
        )
    )
    model = (
        model
        .load_module()
        .Wav2Vec2ForCTC.from_pretrained(repo_id, cache_dir=config.cache_dir)
    )
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

    results = test_set.map(
        evaluate,
        fn_kwargs={
            "model": model,
            "processor": processor,
            "device": config.device,
            "with_lm": config.with_lm,
        },
    )
    results = results.to_pandas()
    logger.info("Evaluation done")

    logger.info(f"WER: {results['wer'].mean():.2f}")
    logger.info(f"CER: {results['cer'].mean():.2f}")

    if config.with_lm:
        results_file = config.output_dir / "results_with_lm.json"
    else:
        results_file = config.output_dir / "results_without_lm.json"
    reduced_results = results[["ref", "hyp", "cer", "wer"]]
    reduced_results.to_json(results_file, default_handler=str)
    logger.info(f"Results saved to {results_file}")
