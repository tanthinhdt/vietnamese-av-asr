import torch
import logging


def check_compatibility_with_bf16(
    compute_dtype: torch.dtype,
    use_4bit: bool,
) -> bool:
    '''
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
    '''
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return True
    return False


def get_logger(name: str, log_path: str = None) -> logging.Logger:
    '''
    Get a logger.

    Parameters
    ----------
    name : str
        Name of the logger
    log_path : str
        Path to the log file

    Returns
    -------
    logging.Logger
        Logger
    '''
    logger = logging.get_logger(name)
    logger.setLevel(logging.INFO)
    logger.addFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logger.addHandler(logging.StreamHandler())
    if log_path is not None:
        logger.addHandler(logging.FileHandler(log_path))
    return logger
