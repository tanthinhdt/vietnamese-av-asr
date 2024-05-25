import os
import shutil
from typing import List


def prepare_dir(dir: list, overwrite: bool = False) -> str:
    """
    Create directory if not exists. If exists, remove it and create new one.
    :param dir:         Path to directory.
    :param overwrite:   Whether to overwrite existing directory.
    :return:            Path to directory.
    """
    if overwrite and os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir, exist_ok=True)
    return dir


def zip_dir(zip_dir: str, overwrite: bool = False) -> str:
    """
    Zip directory.
    :param zip_dir:     Path to directory.
    :param overwrite:   Whether to overwrite existing zip file.
    :return:            Path to zip file.
    """
    if overwrite and os.path.exists(zip_dir + ".zip"):
        os.remove(zip_dir + ".zip")
    shutil.make_archive(
        zip_dir, "zip", os.path.dirname(zip_dir), os.path.basename(zip_dir)
    )
    return zip_dir + ".zip"


def check_num_samples_in_dir(dir_path: str, num_samples: int):
    """
    Check if number of samples in directory matches expected number of samples.
    :param dir:             Path to directory.
    :param num_samples:     Number of samples.
    :raises:                AssertionError if number of samples in directory does not
                            match expected number of samples.
    """
    num_samples_in_dir = len(os.listdir(dir_path))
    print(f"Expected {num_samples} in {dir_path}, but got {num_samples_in_dir}")


def get_file_ids_in_dir(dir: str) -> List[str]:
    """
    Get file ids in directory.
    :param dir:     Path to directory.
    :return:        List of file ids.
    """
    return [file_id.split(".")[0] for file_id in os.listdir(dir)]
