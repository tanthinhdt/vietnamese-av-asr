import torch
import editdistance as ed
from thop import profile
from typing import Tuple


def compute_wer(hyp: str, ref: str) -> float:
    return ed.eval(ref.split(), hyp.split()) * 100 / len(ref.split())


def compute_cer(hyp: str, ref: str) -> float:
    return ed.eval(ref, hyp) * 100 / len(ref)


def compute_flops(model: torch.nn.Module, inputs: Tuple[torch.Tensor]) -> int:
    return profile(model, inputs=inputs)[0]
