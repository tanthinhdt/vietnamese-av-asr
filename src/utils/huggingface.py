import os
import tempfile
from glob import glob
from logging import Logger
from huggingface_hub import HfApi, HfFileSystem, CommitScheduler
from .zipping import zip_dir


os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


class UploadScheduler(CommitScheduler):
    """
    This class is used to upload files to HuggingFace repository.
    """

    def __init__(
        self,
        *,
        logger: Logger,
        glob_pattern: str = None,
        delete_after_upload: bool = False,
        overwrite: bool = False,
        zip: bool = False,
        reverse: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.folder_path = str(self.folder_path)
        if glob_pattern is not None:
            self.folder_path = os.path.join(self.folder_path, glob_pattern)
        self.path_in_repo = str(self.path_in_repo)
        self.logger = logger
        self.delete_after_upload = delete_after_upload
        self.overwrite = overwrite
        self.zip = zip
        self.reverse = reverse
        self.is_done = False
        self.logger.info(f"Created UploadScheduler for {self.repo_id} repo")

    def push_to_hub(self):
        """
        Push files to HuggingFace repository.
        """
        self.logger.info("Start new pushing cycle")
        paths = sorted(glob(self.folder_path), reverse=self.reverse)
        self.logger.info(f"Found {len(paths)} to be pushed")
        if len(paths) == 0:
            self.is_done = True
            return

        for i, path in enumerate(paths):
            dest_path = os.path.join(
                self.path_in_repo, os.path.basename(path)
            )
            if os.path.isdir(path) and self.zip:
                dest_path += ".zip"
            if not self.overwrite and exist_in_hf(
                repo_id=self.repo_id,
                path_in_repo=dest_path,
                repo_type=self.repo_type,
            ):
                continue
            self.logger.info(f"[{i + 1}/{len(paths)}] Uploading {path}")

            if os.path.isdir(path) and self.zip:
                with tempfile.TemporaryDirectory() as temp_dir:
                    src_path = os.path.join(temp_dir, os.path.basename(path))
                    zip_dir(
                        dir_path=path,
                        output_dir=temp_dir,
                        logger=self.logger,
                    )
                    src_path = src_path + ".zip"
            else:
                src_path = path

            upload_to_hf(
                src_path=src_path,
                dest_path=dest_path,
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                logger=self.logger,
            )

            if self.zip and os.path.exists(src_path):
                os.remove(src_path)
                self.logger.info(f"Deleted {src_path}")

            if self.delete_after_upload:
                os.remove(path)
                self.logger.info(f"Deleted {path}")

            if i == len(paths) - 1:
                self.is_done = True


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
