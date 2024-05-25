import os
from dataclasses import dataclass, field
from .file_system import prepare_dir


@dataclass
class TaskConfig:
    """
    This config is used to process data.
    """
    output_dir: str
    channel_names: str = None
    overwrite: bool = False
    upload_to_hub: bool = False
    clean_input: bool = False
    clean_output: bool = False
    version: int = 1
    cache_dir: str = os.path.join(os.getcwd(), ".cache")

    task: str = None
    src_repo_id: str = None
    dest_repo_id: str = None
    schemas: list = None
    num_proc: int = 1
    remove_columns_loading: list = None
    remove_columns_mapping: list = None

    schema_dict: dict = None

    def prepare_dir(self, channel: str, overwrite: bool) -> None:
        """
        Prepare directory.
        :param channel:     Channel name.
        :param overwrite:   Whether to overwrite.
        """
        schemas = dict()
        if self.schemas is not None:
            for schema in self.schemas:
                data_dir = prepare_dir(
                    dir=os.path.join(self.output_dir, schema, channel),
                    overwrite=overwrite,
                )
                schemas[schema] = data_dir
        self.schema_dict = schemas
        return self

    def get_task_kwargs(self) -> dict:
        """
        Get task kwargs.
        :return:            Task kwargs.
        """
        assert self.schema_dict is not None, "Please call prepare_dir() first."
        task_kwargs = {
            "num_proc": self.num_proc,
            "remove_columns": self.remove_columns_mapping,
        }
        fn_kwargs = dict()
        for key, data_dir in self.schema_dict.items():
            fn_kwargs[key + "_output_dir"] = data_dir
        task_kwargs["fn_kwargs"] = fn_kwargs
        return task_kwargs

    def print_num_samples_change(self, num_samples_change: int) -> None:
        """
        Print changes.
        :param num_samples_change:  Number of samples change.
        """
        print(f"\tNumber of samples lost: {num_samples_change}")


@dataclass
class DenoisingTaskConfig(TaskConfig):
    """
    This config is used to denoise audio.
    """
    task: str = "denoise"
    src_repo_id: str = "phdkhanh2507/vietnamese-speaker-clip"
    dest_repo_id: str = "phdkhanh2507/denoised-vietnamese-audio"
    schemas: list = field(default_factory=lambda: ["audio"])
    remove_columns_loading: list = field(default_factory=lambda: ["visual"])
    remove_columns_mapping: list = field(default_factory=lambda: ["audio"])

    def get_task_kwargs(self) -> dict:
        """
        Get task kwargs.
        :return:            Task kwargs.
        """
        task_kwargs = super().get_task_kwargs()
        task_kwargs["fn_kwargs"].update({
            "output_sampling_rate": 16000,
        })
        return task_kwargs
    
    
@dataclass
class TranscribingTaskConfig(TaskConfig):
    """
    This config is used to transcribe audio.
    """
    task: str = "transcribe"
    src_repo_id: str = "phdkhanh2507/denoised-vietnamese-audio"
    dest_repo_id: str = "phdkhanh2507/transcribed-vietnamese-audio"
    # schemas: list = field(default_factory=lambda: ["transcript"])
    # schemas: list = []
    remove_columns_mapping: list = field(default_factory=lambda: ["audio"])

    def get_task_kwargs(self) -> dict:
        """
        Get task kwargs.
        :return:            Task kwargs.
        """
        task_kwargs = super().get_task_kwargs()
        task_kwargs["fn_kwargs"].update({
            "beam_width": 500,
        })
        return task_kwargs


@dataclass
class CroppingTaskConfig(TaskConfig):
    """
    This config is used to crop mouth region in video.
    """
    task: str = "crop"
    src_repo_id: str = "phdkhanh2507/vietnamese-speaker-clip"
    dest_repo_id: str = "phdkhanh2507/vietnamese-speaker-lip-clip"
    schemas: list = field(default_factory=lambda: ["visual"])
    remove_columns_loading: list = field(default_factory=lambda: ["audio"])
    remove_columns_mapping: list = field(default_factory=lambda: ["visual"])

    def get_task_kwargs(self) -> dict:
        """
        Get task kwargs.
        :return:            Task kwargs.
        """
        task_kwargs = super().get_task_kwargs()
        task_kwargs["fn_kwargs"].update({
            "padding": 96,
        })
        return task_kwargs