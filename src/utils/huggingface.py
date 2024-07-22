import os
from glob import glob
from typing import Union
from pathlib import Path
from loguru import logger
from tempfile import TemporaryDirectory
from huggingface_hub import HfApi, HfFileSystem, CommitScheduler
from .zipping import zip_dir
from .misc import delete_file_or_dir


os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


class UploadScheduler(CommitScheduler):
    """
    This class is used to upload files to HuggingFace repository.
    """
    def __init__(
        self,
        *,
        glob_pattern: str = None,
        delete_after_upload: bool = False,
        overwrite: bool = False,
        zip: bool = False,
        reverse: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if glob_pattern is not None:
            self.folder_path = self.folder_path / glob_pattern
        self.delete_after_upload = delete_after_upload
        self.overwrite = overwrite
        self.zip = zip
        self.reverse = reverse
        self.is_done = False
        logger.info(f"Created UploadScheduler for {self.repo_id} repo")

    def push_to_hub(self):
        """
        Push files to HuggingFace repository.
        """
        logger.info("Start new pushing cycle")
        paths = sorted(glob(str(self.folder_path)), reverse=self.reverse)
        logger.info(f"Found {len(paths)} to be pushed")
        if len(paths) == 0:
            self.is_done = True
            return

        for i, path in enumerate(paths):
            path = Path(path)
            dest_path = self.path_in_repo / path.name
            if path.is_dir() and self.zip:
                dest_path += ".zip"
                zipping = True
            else:
                zipping = False
            if not self.overwrite and exists_on_hf(
                repo_id=self.repo_id,
                path_in_repo=dest_path,
                repo_type=self.repo_type,
            ):
                continue
            logger.info(f"[{i + 1}/{len(paths)}] Uploading {path}")

            if zipping:
                with TemporaryDirectory() as temp_dir:
                    temp_dir = Path(temp_dir)
                    src_path = temp_dir / path.name
                    zip_dir(
                        dir_path=path,
                        output_dir=temp_dir,
                    )
                    src_path = src_path + ".zip"
                    upload_to_hf(
                        src_path=src_path,
                        dest_path=dest_path,
                        repo_id=self.repo_id,
                        repo_type=self.repo_type,
                    )
                    os.remove(src_path)
                    logger.info(f"Deleted {src_path}")
            else:
                src_path = path
                upload_to_hf(
                    src_path=src_path,
                    dest_path=dest_path,
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                )

            if self.delete_after_upload:
                delete_file_or_dir(path)
                logger.info(f"Deleted {path}")

        self.is_done = True


def upload_to_hf(
    src_path: Union[str, Path],
    dest_path: Union[str, Path],
    repo_id: str,
    repo_type: str,
) -> None:
    """
    Upload a file or directory to HuggingFace repository.

    Parameters
    ----------
    src_path : Union[str, Path]
        Path to the file or directory to upload.
    dest_path : Union[str, Path]
        Path to upload in the repository.
    repo_id : str
        HuggingFace repository ID.
    repo_type : str
        Type of the repository.
    """
    api = HfApi()
    dest_path = str(dest_path)
    if Path(src_path).is_dir():
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


def get_hf_path(
    repo_id: str,
    path_in_repo: Union[str, Path],
    repo_type: str,
) -> Path:
    """
    Get the path in the HuggingFace repository.

    Parameters
    ----------
    repo_id : str
        HuggingFace repository ID.
    path_in_repo : Union[str, Path]
        Path to check in the repository.
    repo_type : str
        Type of the repository.

    Returns
    -------
    Path
        Path in the repository.
    """
    if repo_type == "dataset":
        hf_path = Path(f"datasets/{repo_id}")
    elif repo_type == "model":
        hf_path = Path(repo_id)
    else:
        raise ValueError("repo_type should be either 'dataset' or 'model'")
    return hf_path / path_in_repo


def exists_on_hf(
    repo_id: str,
    path_in_repo: Union[str, Path],
    repo_type: str,
) -> bool:
    """
    Check if a file or directory exists in HuggingFace repository.

    Parameters
    ----------
    repo_id : str
        HuggingFace repository ID.
    path_in_repo : Union[str, Path]
        Path to check in the repository.
    repo_type : str
        Type of the repository.

    Returns
    -------
    bool
        True if the file or directory exists.
    """
    hf_path = get_hf_path(repo_id, path_in_repo, repo_type)
    fs = HfFileSystem()
    return fs.exists(hf_path)


def get_paths(path: Union[Path, str]) -> list:
    """
    Get all paths in the HuggingFace repository.

    Parameters
    ----------
    path : Union[Path, str]
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
    path_in_repo: Union[str, Path],
    output_dir: Union[str, Path],
    overwrite: bool = False,
    repo_type: str = "dataset",
) -> Path:
    """
    Download files from HuggingFace repository.

    Parameters
    ----------
    repo_id : str
        HuggingFace repository ID.
    path_in_repo : Union[str, Path]
        Path to download in the repository.
    output_dir : Union[str, Path]
        Output directory.
    overwrite : bool, False
        Overwrite existing files.
    repo_type : str, 'dataset'
        Type of the repository.

    Returns
    -------
    Path
        Path to the downloaded file or directory.
    """
    fs = HfFileSystem()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hf_root_path = get_hf_path(repo_id, path_in_repo, repo_type)
    hf_paths = get_paths(hf_root_path)

    for i, (root, file) in enumerate(hf_paths):
        root = Path(root)
        if file is None:
            file_output_dir = output_dir
            file, root = root.name, root.parent
        else:
            file_output_dir = output_dir / root.relative_to(hf_root_path.name)
        logger.info(f"[{i + 1}/{len(hf_paths)}] Processing {root}/{file}")

        file_output_dir.mkdir(parents=True, exist_ok=True)

        if overwrite or not (file_output_dir / file).exists():
            try:
                fs.download(str(root / file), str(file_output_dir))
                logger.info(f"\tDownloaded to {file_output_dir}")
            except KeyboardInterrupt:
                os.remove(file_output_dir / file)
                raise KeyboardInterrupt
        else:
            logger.info(f"\tFile exists in {file_output_dir}")

    return output_dir / Path(path_in_repo).name
