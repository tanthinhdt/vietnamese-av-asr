import os
import torch
import shutil
from pathlib import Path
from typing import Union


def check_compatibility_with_bf16(
    compute_dtype: torch.dtype,
    use_4bit: bool,
) -> bool:
    """
    Check if the GPU supports bfloat16.

    Parameters
    ----------
    compute_dtype : torch.dtype
        Data type for computation
    use_4bit : bool
        Use 4-bit quantization

    Returns
    -------
    bool
        True if the GPU supports bfloat16
    """
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return True
    return False


def delete_file_or_dir(path: Union[str, Path]) -> None:
    """
    Remove a file or directory.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the file or directory to remove.
    """
    if Path(path).is_dir():
        shutil.rmtree(path)
    else:
        os.remove(path)
