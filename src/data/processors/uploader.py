import os
from huggingface_hub import HfApi, HfFolder, HfFileSystem
from src.data.utils import zip_dir
from .processor import Processor


class Uploader(Processor):
    TOKEN = HfFolder.get_token()

    def __init__(self) -> None:
        """
        Initialize processor.
        """
        self.api = HfApi()
        self.fs = HfFileSystem()

    def zip_and_upload_dir(
        self, dir_path: str,
        repo_id: str,
        path_in_repo: str,
        repo_type: str = "dataset",
        overwrite: bool = True,
    ) -> None:
        """
        Zip directory and upload it to the hub.

        dir_path:     
            Path to directory.
        repo_id:
            Repository id.
        path_in_repo:
            Path to directory in repository.
        repo_type:
            Repository type.
        """
        if overwrite or not self.fs.exists(f"{repo_type}s/{repo_id}/{path_in_repo}"):
            self.api.upload_file(
                path_or_fileobj=zip_dir(dir_path, overwrite=True),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                token=self.TOKEN,
                repo_type=repo_type,
            )
            os.remove(dir_path + ".zip")

    def upload_file(
        self, file_path: str,
        repo_id: str,
        path_in_repo: str,
        repo_type: str = "dataset",
        overwrite: bool = True,
    ) -> None:
        """
        Upload file to the hub.

        file_path:
            Path to file.
        repo_id:
            Repository id.
        path_in_repo:
            Path to file in repository.
        repo_type:
            Repository type.
        """
        if overwrite or not self.fs.exists(f"{repo_type}s/{repo_id}/{path_in_repo}"):
            self.api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                token=self.TOKEN,
                repo_type=repo_type,
            )
