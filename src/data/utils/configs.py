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
    cache_dir: str = os.path.join(os.getcwd(), "data" , "external")

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
class VideoDownloadTaskConfig(TaskConfig):
    task: str = 'download'
    src_repo_id: str = 'GSU24AI03-SU24AI21/tracked-url-video'
    dest_repo_id: str = 'GSU24AI03-SU24AI21/downloaded-vietnamese-video'
    schemas: list = field(default_factory=lambda: ['video'])
    remove_columns_loading: list = field(default_factory=lambda: [])
    remove_columns_mapping: list = field(default_factory=lambda: ['url'])

    def get_task_kwargs(self) -> dict:
        """
        Get task keyword arguments.
        :return:
        """
        task_kwargs = super().get_task_kwargs()
        task_kwargs["fn_kwargs"].update({
            'log_path': 'src/data/databases/logs/yt-downloader.log',
        })
        return task_kwargs


@dataclass
class SpeakerDetectTaskConfig(TaskConfig):
    task: str = 'asd'
    src_repo_id: str = 'GSU24AI03-SU24AI21/downloaded-vietnamese-video'
    dest_repo_id: str = 'GSU24AI03-SU24AI21/detected-speaker-clip'
    schemas: list = field(default_factory=lambda: ['visual', 'audio'])
    remove_columns_loading: list = field(default_factory=lambda: ["duration", "video_fps", "audio_fps"])
    remove_columns_mapping: list = field(default_factory=lambda: ["video_path", "video_id"])

    def get_task_kwargs(self) -> dict:
        """
        Get task keyword arguments.
        :return:
        """
        task_kwargs = super().get_task_kwargs()
        task_kwargs["fn_kwargs"].update({
            "output_dir": self.output_dir,
            "tmp_dir": os.path.join(os.getcwd(), "data", "interim"),
            'log_path': 'src/data/databases/logs/as-detector.log',
        })
        return task_kwargs


@dataclass
class MouthCropTaskConfig(TaskConfig):
    """
    This config is used to crop mouth region in video.
    """
    task: str = "crop"
    src_repo_id: str = "GSU24AI03-SU24AI21/detected-speaker-clip"
    dest_repo_id: str = "GSU24AI03-SU24AI21/cropped-mouth-clip"
    schemas: list = field(default_factory=lambda: ["visual"])
    remove_columns_loading: list = field(default_factory=lambda: [])
    remove_columns_mapping: list = field(default_factory=lambda: ["visual_path"])

    def get_task_kwargs(self) -> dict:
        """
        Get task kwargs.
        :return:            Task kwargs.
        """
        task_kwargs = super().get_task_kwargs()
        task_kwargs["fn_kwargs"].update({
            "padding": 96,
            "log_path": 'src/data/databases/logs/cropper.log',
        })
        return task_kwargs

@dataclass
class VietnameseDetectTaskConfig(TaskConfig):
    task: str = 'vndetect'
    src_repo_id: str = 'GSU24AI03-SU24AI21/cropped-mouth-clip'
    dest_repo_id: str = 'GSU24AI03-SU24AI21/detected-vietnamese-clip'
    schemas: list = field(default_factory=lambda: ['audio'])
    remove_columns_loading: list = field(default_factory=lambda: [])
    remove_columns_mapping: list = field(default_factory=lambda: ["audio_path"])

    def get_task_kwargs(self) -> dict:
        """
        Get task keyword arguments.
        :return:
        """
        task_kwargs = super().get_task_kwargs()
        task_kwargs["fn_kwargs"].update({
            "log_path": 'src/data/databases/logs/vn-detector.log',
        })
        return task_kwargs


@dataclass
class TranscribeTaskConfig(TaskConfig):
    """
    This config is used to transcribe audio.
    """
    task: str = "transcribe"
    src_repo_id: str = "GSU24AI03-SU24AI21/detected-vietnamese-clip"
    dest_repo_id: str = "GSU24AI03-SU24AI21/transcribed-vietnamese-audio"
    schemas: list = field(default_factory=lambda: [])
    remove_columns_loading: list = field(default_factory=lambda: [])
    remove_columns_mapping: list = field(default_factory=lambda: ["audio_path"])

    def get_task_kwargs(self) -> dict:
        """
        Get task kwargs.
        :return:            Task kwargs.
        """
        task_kwargs = super().get_task_kwargs()
        task_kwargs["fn_kwargs"].update({
            "beam_width": 500,
            "log_path": 'src/data/databases/logs/transcriber.log',
        })
        return task_kwargs