import itertools
from typing import List
from loguru import logger


def load_label(label_path: str, inds: List[int], tot: int) -> List[str]:
    with open(label_path) as f:
        labels = [line.rstrip() for line in f]
        assert len(labels) == tot, \
            f"number of labels does not match ({len(labels)} != {tot})"
        labels = [labels[i] for i in inds]
    return labels


def load_label_offset(label_path: str, inds: List[int], tot: int) -> List[str]:
    with open(label_path) as f:
        code_lengths = [len(line.encode("utf-8")) for line in f]
        assert len(code_lengths) == tot, \
            f"number of labels does not match ({len(code_lengths)} != {tot})"
        offsets = list(itertools.accumulate([0] + code_lengths))
        offsets = [(offsets[i], offsets[i + 1]) for i in inds]
    return offsets


def verify_label_lengths(
    audio_sizes: List[int],
    audio_rate: float,
    label_path: str,
    label_rate: int,
    inds: List[int],
    tot: int,
    tol: float = 0.1,  # tolerance in seconds
) -> None:
    if label_rate < 0:
        logger.info(f"{label_path} is sequence label. skipped")
        return

    with open(label_path) as f:
        lengths = [len(line.rstrip().split()) for line in f]
        assert len(lengths) == tot
        lengths = [lengths[i] for i in inds]
    num_invalid = 0
    for i, ind in enumerate(inds):
        dur_from_audio = audio_sizes[i] / audio_rate
        dur_from_label = lengths[i] / label_rate
        if abs(dur_from_audio - dur_from_label) > tol:
            logger.warning(
                (
                    f"audio and label duration differ too much "
                    f"(|{dur_from_audio} - {dur_from_label}| > {tol}) "
                    f"in line {ind+1} of {label_path}. Check if `label_rate` "
                    f"is correctly set (currently {label_rate}). "
                    f"num. of samples = {audio_sizes[i]}; "
                    f"label length = {lengths[i]}"
                )
            )
            num_invalid += 1
    if num_invalid > 0:
        logger.warning(
            f"total {num_invalid} (audio, label) pairs with mismatched lengths"
        )
