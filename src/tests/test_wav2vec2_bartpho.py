import re
import torch
from transformers import (
    AutoFeatureExtractor,
    AutoTokenizer,
    SpeechEncoderDecoderModel,
    GenerationConfig,
)
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
    processor: AutoFeatureExtractor,
    tokenizer: AutoTokenizer,
    data: torch.Tensor,
    device: str,
    prefix: str = "",
) -> torch.Tensor:
    audio_wavs = [torch.from_numpy(data)]

    input_values = processor.pad(
        [{"input_values": audio} for audio in audio_wavs],
        padding=True,
        max_length=None,
        pad_to_multiple_of=None,
        return_tensors="pt",
    )

    decoder_input_ids = tokenizer.batch_encode_plus(
        [prefix] * len(audio_wavs),
        return_tensors="pt",
    )
    decoder_input_ids = decoder_input_ids['input_ids'][..., :-1]
    input_values["decoder_input_ids"] = decoder_input_ids

    input_values = {k: v.to(device) for k, v in input_values.items()}
    return input_values


def decode_tokens(
    tokenizer: AutoTokenizer,
    token_ids: torch.Tensor,
    skip_special_tokens: bool = True,
) -> str:
    timestamp_begin = tokenizer.vocab_size
    outputs = []

    for token in token_ids:
        if token < timestamp_begin:
            outputs.append(token)

    sequence = tokenizer.decode(
        outputs,
        skip_special_tokens=skip_special_tokens,
    )
    sequence = re.sub(r"<", "", sequence)
    sequence = re.sub(r">", "", sequence)
    sequence = re.sub(r"\s+", " ", sequence)
    sequence = sequence.strip()

    return sequence


def evaluate(
    sample: dict,
    model: SpeechEncoderDecoderModel,
    processor: AutoFeatureExtractor,
    tokenizer: AutoTokenizer,
    device: str,
) -> dict:
    input_values = get_input_values(
        tokenizer=tokenizer,
        processor=processor,
        data=sample["audio"]["array"],
        device=device,
    )

    output_beam_ids = model.generate(
        input_values["input_values"],
        attention_mask=input_values["attention_mask"],
        decoder_input_ids=input_values["decoder_input_ids"],
        generation_config=GenerationConfig(decoder_start_token_id=tokenizer.bos_token_id),
        max_length=250,
        num_beams=25,
        no_repeat_ngram_size=4,
        num_return_sequences=1,
        early_stopping=True,
        return_dict_in_generate=True,
        output_scores=True,
    )
    sequence = output_beam_ids.sequences[0]
    sample["hyp"] = decode_tokens(tokenizer, sequence)

    sample["wer"] = compute_wer(sample["hyp"], sample["ref"])
    sample["cer"] = compute_cer(sample["hyp"], sample["ref"])

    return sample


def test_wav2vec2_bartpho(config: TestConfig) -> None:
    test_set = load_test_set(config.test_manifest_file, config.test_ref_file)

    repo_id = "nguyenvulebinh/wav2vec2-bartpho"
    logger.info(f"Model {repo_id}")

    processor = AutoFeatureExtractor.from_pretrained(
        repo_id,
        cache_dir=config.cache_dir,
    )
    logger.info("Processor loaded")

    tokenizer = AutoTokenizer.from_pretrained(
        repo_id,
        cache_dir=config.cache_dir,
    )
    logger.info("Tokenizer loaded")

    model = SpeechEncoderDecoderModel.from_pretrained(
        repo_id,
        cache_dir=config.cache_dir,
    )
    model = model.eval().to(config.device)
    logger.info("Model loaded")
    log_params(model)

    sample = test_set[0]["audio"]
    input_values = get_input_values(
        tokenizer=tokenizer,
        processor=processor,
        data=sample["array"],
        device=config.device,
    )
    flops = compute_flops(
        model,
        inputs=(
            input_values["input_values"],
            input_values["attention_mask"],
            input_values["decoder_input_ids"],
        )
    )
    logger.info(f"FLOPs: {flops:,}")
    logger.info(f"FLOPs (G): {flops / 1_000_000_000:.2f}")

    results = test_set.map(
        evaluate,
        fn_kwargs={
            "model": model,
            "processor": processor,
            "tokenizer": tokenizer,
            "device": config.device,
        },
    )
    results = results.to_pandas()
    logger.info("Evaluation done")
    log_average_results(results)

    results_file = config.output_dir / "results.json"
    save_results(results, results_file)
