from pathlib import Path
from typing import Any, List
from dataclasses import dataclass, field
from utils import MODELS


EXTENSIONS = (
    ".mp4", ".avi", ".mov", ".mkv",
    ".mp3", ".wav", ".flac", ".ogg"
)


@dataclass
class ProcessConfig:
    path: str = None    # Path to the file or directory to process
    log_path: str = None    # Path to the log file
    overwrite: bool = False     # Overwrite existing files
    reverse: bool = False   # Reverse the order of the files

    def __post_init__(self):
        self.path = Path(self.path)
        assert self.path.exists(), f"Path {self.path} does not exist."
        if self.log_path is not None:
            self.log_path = Path(self.log_path)


@dataclass
class UploadConfig(ProcessConfig):
    repo_id: str = None     # HuggingFace repository ID
    dir_in_repo: str = "."      # Path to upload in the repository
    repo_type: str = "dataset"      # Type of the repository
    every_minutes: float = 1    # Upload every F minutes
    delete_after_upload: bool = False   # Delete the file or directory after uploading
    zip: bool = False   # Zip the directory before uploading

    def __post_init__(self):
        super().__post_init__()
        assert self.repo_id is not None, "Repository ID is required"
        self.dir_in_repo = Path(self.dir_in_repo)
        assert self.repo_type in ["dataset", "model"], \
            "Only Dataset and Model repositories are supported"
        assert self.every_minutes > 0, "Upload interval should be positive"


@dataclass
class TransformConfig:
    horizontal_flip_prob: float = 0.5
    aug_type: str = "augmix"
    aug_paras: dict = field(
        default_factory=lambda: {
            "magnitude": 3,
            "alpha": 1.0,
            "width": 5,
            "depth": -1,
        }
    )
    sample_rate: int = 4

    def __post_init__(self):
        assert self.aug_type in ["augmix", "mixup"], \
            "Only AugMix and MixUp are supported for now"


@dataclass
class DataConfig:
    dataset: str = "vasr"
    modality: List[str] = field(default_factory=lambda: ["audio", "video"])
    subset: str = None
    data_dir: str = "data/processed/vsl"
    transform: Any = None
    fps: int = 30
    debug: bool = False
    transform: TransformConfig = TransformConfig()

    def __post_init__(self):
        assert self.dataset in ["vasr"], \
            "Only VSL dataset is supported for now"
        assert self.modality in ["audio", "video"], \
            "Only RGB and Pose modalities are supported for now"


@dataclass
class ModelConfig:
    arch: str = "avsp_llm"
    pretrained: str = "DEFAULT"
    num_frozen_layers: int = 0
    ignored_weights: List[str] = field(default_factory=lambda: [])
    num_frames: int = 16

    def __post_init__(self):
        assert self.arch in MODELS, f"Model {self.arch} is not supported"


@dataclass
class TrainConfig:
    output_dir: str = "experiments"
    remove_unused_columns: bool = False
    do_train: bool = True
    use_cpu: bool = False

    eval_strategy: str = "epoch"
    logging_strategy: str = "epoch"
    save_strategy: str = "epoch"
    logging_steps: int = 1
    save_steps: int = 1
    eval_steps: int = 1

    learning_rate: float = 5e-5
    weight_decay: float = 0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    warmup_ratio: float = 0.1

    num_train_epochs: int = 10
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    dataloader_num_workers: int = 0

    load_best_model_at_end: bool = True
    metric_for_best_model: str = "accuracy"
    resume_from_checkpoint: str = None

    run_name: str = "swin3d"
    report_to: str = None
    push_to_hub: bool = False
    hub_model_id: str = None
    hub_strategy: str = "checkpoint"
    hub_private_repo: bool = True

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        if str(self.output_dir) == "experiments":
            self.output_dir = self.output_dir / self.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.hub_model_id is not None:
            self.push_to_hub = True
            if len(self.hub_model_id.split("/")) == 1:
                self.hub_model_id = f"{self.hub_model_id}/{self.run_name}"


@dataclass
class InferenceConfig:
    source: str = "webcam"
    output_dir: str = "demo"
    use_onnx: bool = False
    device: str = "cpu"
    cache_dir: str = "models/huggingface"

    visualize: bool = False
    show_skeleton: bool = False

    def __post_init__(self):
        self.source = Path(self.source)
        assert any((
            str(self.source) == "webcam",
            (self.source.exists() and str(self.source).endswith(EXTENSIONS))
        )), \
            f"Only Webcam and Video sources are supported for now (got {self.source})"
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
