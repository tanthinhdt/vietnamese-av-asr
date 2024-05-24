import os
from logging import Logger
from huggingface_hub import HfApi, HfFileSystem


def upload_to_hf(
    src_path: str,
    dest_path: str,
    repo_id: str,
    repo_type: str,
    logger: Logger,
) -> None:
    """
    Upload a file or directory to HuggingFace repository

    Parameters
    ----------
    src_path : str
        Path to the file or directory to upload
    dest_path : str
        Path to upload in the repository
    repo_id : str
        HuggingFace repository ID
    repo_type : str
        Type of the repository
    """
    api = HfApi()
    if os.path.isdir(src_path):
        api.upload_folder(
            repo_id=repo_id,
            folder_path=src_path,
            path_in_repo=dest_path,
            repo_type=repo_type,
        )
    else:
        api.upload_file(
            path_or_fileobj=src_path,
            path_in_repo=dest_path,
            repo_id=repo_id,
            repo_type=repo_type,
        )
    logger.info(f"Uploaded {src_path} to {repo_id} repository")


def exist_in_hf(
    repo_id: str,
    path_in_repo: str,
    repo_type: str,
) -> bool:
    """
    Check if a file or directory exists in HuggingFace repository

    Parameters
    ----------
    repo_id : str
        HuggingFace repository ID
    path_in_repo : str
        Path to check in the repository
    repo_type : str
        Type of the repository

    Returns
    -------
    bool
        True if the file or directory exists
    """
    if repo_type == "dataset":
        hf_path = f"datasets/{repo_id}/{path_in_repo}"
    elif repo_type == "model":
        hf_path = f"{repo_id}/{path_in_repo}"
    else:
        raise ValueError("repo_type should be either 'dataset' or 'model'")
    fs = HfFileSystem()
    return fs.exists(hf_path)


def get_paths(path: str) -> list:
    """
    Get all paths in the HuggingFace repository.

    Parameters
    ----------
    path : str
        Path to the repository.

    Returns
    -------
    list
        List of paths in the repository.
    """
    fs = HfFileSystem()
    if fs.isfile(path):
        return [(path, None)]
    hf_paths = []
    for root, _, files in fs.walk(path):
        for file in files:
            hf_paths.append((root, file))
    return hf_paths


def download_from_hf(
    repo_id: str,
    path_in_repo: str,
    output_dir: str,
    logger: Logger,
    overwrite: bool = False,
    repo_type: str = "dataset",
):
    """
    Download files from HuggingFace repository.

    Parameters
    ----------
    repo_id : str
        HuggingFace repository ID.
    path_in_repo : str
        Path to download in the repository.
    output_dir : str
        Output directory.
    overwrite : bool, False
        Overwrite existing files.
    repo_type : str, 'dataset'
        Type of the repository.
    """
    if repo_type == "dataset":
        hf_root_path = f"datasets/{repo_id}/{path_in_repo}"
    elif repo_type == "model":
        hf_root_path = f"{repo_id}/{path_in_repo}"
    else:
        logger.error("repo_type should be either 'dataset' or 'model'")
        return

    fs = HfFileSystem()
    os.makedirs(output_dir, exist_ok=True)
    hf_paths = get_paths(hf_root_path)
    hf_root_path = os.path.normpath(hf_root_path)
    for i, (root, file) in enumerate(hf_paths):
        if file is None:
            file_output_dir = output_dir
            file = os.path.basename(root)
            root = os.path.dirname(root)
        else:
            file_output_dir = os.path.join(
                output_dir,
                os.path.relpath(root, os.path.dirname(hf_root_path)),
            )
        logger.info(f"[{i + 1}/{len(hf_paths)}] Processing {root}/{file}")

        if not os.path.exists(file_output_dir):
            os.makedirs(file_output_dir, exist_ok=True)

        if overwrite or not os.path.exists(os.path.join(file_output_dir, file)):
            try:
                fs.download(
                    f"{root}/{file}",
                    file_output_dir,
                    verbose=False,
                )
                logger.info(f"\tDownloaded to {file_output_dir}")
            except KeyboardInterrupt:
                logger.info("\tInterrupted by user")
                os.remove(os.path.join(file_output_dir, file))
                exit()
        else:
            logger.info(f"\tFile exists in {file_output_dir}")
