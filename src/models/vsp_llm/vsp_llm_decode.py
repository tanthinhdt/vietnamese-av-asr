import json
import os

import torch
import numpy as np

from fairseq import utils
from fairseq import tasks
from fairseq.logging import progress_bar
from src.models.utils.logging import get_logger

logger = get_logger("Producing predictions", is_stream=True)


def produce_predictions(
        cfg,
        model,
        saved_cfg,
        llm_tokenizer,
        modalities,
):
    use_cuda = torch.cuda.is_available()

    task = tasks.setup_task(saved_cfg.task)
    task.build_tokenizer(saved_cfg.tokenizer)
    task.build_bpe(saved_cfg.bpe)
    if cfg.common.seed is not None:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    task.cfg.modalities = modalities
    task.cfg.noise_prob = cfg.override.noise_prob
    task.cfg.noise_snr = cfg.override.noise_snr
    task.cfg.noise_wav = cfg.override.noise_wav
    task.cfg.data = cfg.override.data
    task.cfg.label_dir = cfg.override.label_dir
    task.cfg.labels = cfg.override.labels
    task.cfg.label_rate = cfg.override.label_rate

    task.load_dataset('test')
    logger.info('Loaded dataset')

    cfg.dataset.batch_size = 1
    cfg.dataset.max_tokens = 1000
    itr = task.get_batch_iterator(
        dataset=task.dataset(cfg.dataset.gen_subset),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions([None]),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )

    result_dict = {'utt_id': [], 'ref': [], 'hypo': [], 'instruction': []}
    logger.info("Iterate samples to predict")
    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if "net_input" not in sample:
            continue

        if sample['net_input']['source']['video'] is not None:
            sample['net_input']['source']['video'] = sample['net_input']['source']['video'].to(torch.half)
        if sample['net_input']['source']['audio'] is not None:
            sample['net_input']['source']['audio'] = sample['net_input']['source']['audio'].to(torch.half)

        best_hypo = model.generate(
            num_beams=cfg.generation.beam,
            length_penalty=cfg.generation.lenpen,
            **sample["net_input"]
        )
        best_hypo = llm_tokenizer.batch_decode(
            best_hypo, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for i in range(len(sample["id"])):
            result_dict['utt_id'].append(sample['utt_id'][i])
            hypo_str = best_hypo[i]
            result_dict['hypo'].append(hypo_str)
            logger.info(f"HYP: {hypo_str}")

    os.makedirs(cfg.common_eval.results_path, exist_ok=True)
    result_fn = f"{cfg.common_eval.results_path}/hypo.json"
    json.dump(result_dict, open(result_fn, 'w'), indent=4)

    return result_fn