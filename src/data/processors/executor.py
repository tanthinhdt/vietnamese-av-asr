import os
import shutil
from .processor import Processor
from .youtube_downloader import YoutTubeDownloader
from .as_extracter import ActiveSpeakerExtracter
from .vietnamese_detector import VietnameseDetector
from .transcriber import Transcriber
from .cropper import Cropper
from .uploader import Uploader
from datasets import (Dataset, disable_progress_bar, enable_progress_bar, # type: ignore
                      get_dataset_config_names, load_dataset)
from huggingface_hub import HfFileSystem # type: ignore
from src.data.utils import TaskConfig, prepare_dir, check_num_samples_in_dir


class Executor(Processor):
    """
    This processor is used to execute other processors.
    """
    PROCESSORS = {
        "download": YoutTubeDownloader,
        "asd": ActiveSpeakerExtracter,
        "crop": Cropper,
        "vndetect": VietnameseDetector,
        "transcribe": Transcriber,
    }

    def __init__(self, configs: TaskConfig) -> None:
        """
        :param configs:     Task configs.
        """
        self.configs = configs
        self.processor: Processor = self.PROCESSORS[self.configs.task]()
        self.uploader = Uploader()

        self.metadata_dir = prepare_dir(os.path.join(self.configs.output_dir, "metadata"))

        self.dataset: Dataset = None

        self.available_channels = self.__load_channels()
        
    def __load_channels(self) -> list:
        """
        Load channels to process.
        :return:    List of channels to process.
        """
        # Get available channel names.
        available_channels = set(get_dataset_config_names(self.configs.src_repo_id)) - {"all"}
        upload_only = not self.configs.overwrite and self.configs.upload_to_hub
        if not self.configs.overwrite or upload_only:
            existing_channels = set(get_dataset_config_names(self.configs.dest_repo_id)) - {"all"}
            available_channels -= existing_channels
        # Get channel names to process.
        if self.configs.channel_names:
            if os.path.isfile(self.configs.channel_names):
                with open(self.configs.channel_names, "r") as f:
                    new_channels = set(f.read().split())
            else:
                new_channels = {self.configs.channel_names}
            available_channels = available_channels.intersection(new_channels)

        return list(available_channels)

    def prepare_dir(self, channel: str) -> None:
        """
        Prepare directory.
        :param channel:     Channel name.
        """
        self.configs = self.configs.prepare_dir(
            channel=channel, overwrite=self.configs.overwrite
        )

    def load_dataset(self, channel: str) -> Processor:
        """
        Load dataset.
        :param channel:     Channel name.
        :return:            Executor.
        """
        disable_progress_bar()
        self.dataset = load_dataset(
            self.configs.src_repo_id, channel,
            split="train",
            cache_dir=self.configs.cache_dir,
            trust_remote_code=True,
        )
        if self.configs.remove_columns_loading:
            self.dataset = self.dataset.remove_columns(
                self.configs.remove_columns_loading
            )
        enable_progress_bar()

        self.num_samples_before = self.dataset.num_rows
        self.num_samples_after = 0
        return self

    def process(self) -> Processor:
        """
        Process sample.
        :return:                    Executor.
        """
        assert self.dataset is not None, "Dataset is not loaded yet."

        task_kwargs = self.configs.get_task_kwargs()
        self.dataset = self.dataset.map(
            self.processor.process,
            fn_kwargs=task_kwargs["fn_kwargs"],
            batched=True, batch_size=1,
            num_proc=task_kwargs["num_proc"],
            remove_columns=task_kwargs["remove_columns"],
            load_from_cache_file=not self.configs.overwrite,
        )
        disable_progress_bar()
        self.dataset = self.dataset.filter(
            lambda sample: sample["id"] is not None,
            num_proc=os.cpu_count(),
            load_from_cache_file=not self.configs.overwrite,
        )

        enable_progress_bar()
        self.num_samples_after = self.dataset.num_rows
        return self

    def check_num_samples_in_dir(self) -> None:
        """
        Check if number of samples in directory matches expected number of samples.
        """
        assert self.dataset is not None, "Dataset is not loaded yet."

        for data_dir in self.configs.schema_dict.values():
            check_num_samples_in_dir(
                dir_path=data_dir,
                num_samples=self.num_samples_after,
            )

    def print_num_samples_change(self):
        """
        Get number of samples lost.
        """
        self.configs.print_num_samples_change(
            abs(self.num_samples_after - self.num_samples_before)
        )

    def print_num_output_samples(self) -> None:
        """
        Print number of output samples.
        """
        print(f"\tNumber of output samples: {self.num_samples_after}")

    def save_metadata(self, channel: str) -> None:
        """
        Save metadata as parquet file and save channel name to file.
        :param channel:     Channel name.
        """
        assert self.dataset is not None, "Dataset is not loaded yet."

        metadata_path = os.path.join(self.metadata_dir, channel + ".parquet")
        disable_progress_bar()
        self.dataset.to_parquet(metadata_path)
        enable_progress_bar()

    def upload_to_hub(self, channel: str) -> None:
        """
        Upload to hub.
        :param channel:     Channel name.
        """
        if self.configs.upload_to_hub:
            print("Uploading to hub...")
            for schema, data_dir in self.configs.schema_dict.items():
                self.__zip_and_upload_dir(
                    dir_path=data_dir,
                    path_in_repo=os.path.join(schema, channel + ".zip"),
                )
                if self.configs.clean_output and os.path.exists(data_dir):
                    shutil.rmtree(data_dir, ignore_errors=True)
                if self.configs.clean_input and os.path.exists(self.configs.cache_dir):
                    shutil.rmtree(self.configs.cache_dir, ignore_errors=True)
            self.__upload_metadata_to_hub(channel=channel)
            print()

    def __upload_metadata_to_hub(self, channel: str) -> None:
        """
        Upload metadata and channel names to hub.
        :param channel:     Channel name.
        :param overwrite:   Whether to overwrite existing file.
        """
        metadata_path = os.path.join(self.metadata_dir, channel + ".parquet")
        self.uploader.upload_file(
            file_path=metadata_path,
            repo_id=self.configs.dest_repo_id,
            path_in_repo=os.path.join("metadata", channel + ".parquet"),
        )

    def __zip_and_upload_dir(
        self, dir_path: str,
        path_in_repo: str,
    ) -> None:
        """
        Zip directory and upload it to the hub.
        :param dir_path:        Path to directory.
        :param path_in_repo:    Path to directory in repository.
        :param overwrite:       Whether to overwrite existing file.
        """
        self.uploader.zip_and_upload_dir(
            dir_path=dir_path,
            repo_id=self.configs.dest_repo_id,
            path_in_repo=path_in_repo,
        )

    def is_skipped(self, channel) -> bool:
        fs = HfFileSystem()
        path_in_hub = os.path.join(
            "datasets", self.configs.dest_repo_id, "metadata", f"{channel}.parquet"
        )
        if not fs.exists(path_in_hub):
            return False
        if self.configs.overwrite:
            return False
        return True
