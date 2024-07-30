from glob import glob
from pathlib import Path
from dataclasses import dataclass, field


EXTENSIONS = (
    ".mp4", ".avi", ".mov", ".mkv",
    ".mp3", ".wav", ".flac", ".ogg"
)


@dataclass
class ProcessConfig:
    ...


@dataclass
class CreateManifestConfig(ProcessConfig):
    data_dir: str = None
    split: str = None
    src_lang: str = "vi"
    dst_lang: str = "vi"
    frac: float = 1.0
    output_dir: str = None

    def __post_init__(self):
        assert self.data_dir is not None, "Data directory is required"
        self.data_dir = Path(self.data_dir)
        assert self.data_dir.exists(), f"Data directory {self.data_dir} is not found"
        self.metadata_path = self.data_dir / "metadata.parquet"
        assert self.metadata_path.exists(), "Metadata file is not found"
        assert self.split is not None, "Split is required"
        assert 0 < self.frac <= 1.0, "Fraction should be in (0, 1]"
        assert self.output_dir is not None, "Output directory is required"
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class DumpFeatureConfig(ProcessConfig):
    tsv_dir: str = None
    split: str = None
    ckpt_path: str = None
    layer: int = None
    nshard: int = None
    rank: int = None
    feat_dir: str = None
    max_chunk: int = 1_600_000

    def __post_init__(self):
        assert self.tsv_dir is not None, "TSV directory is required"
        self.tsv_dir = Path(self.tsv_dir)
        assert self.split is not None, "Split is required"
        assert self.ckpt_path is not None, "Checkpoint path is required"
        self.ckpt_path = Path(self.ckpt_path)
        assert self.layer is not None, "Layer is required"
        assert self.nshard is not None, "Number of shards is required"
        assert self.rank is not None, "Rank is required"
        assert self.feat_dir is not None, "Feature directory is required"
        self.feat_dir = Path(self.feat_dir)
        self.feat_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class LearnKMeansConfig(ProcessConfig):
    feat_dir: str = None
    split: str = None
    nshard: int = None
    km_path: str = None
    n_clusters: int = None
    seed: int = 0
    percent: float = -1
    init: str = "k-means++"
    max_iter: int = 100
    batch_size: int = 10_000
    tol: float = 0.0
    max_no_improvement: int = 100
    n_init: int = 20
    reassignment_ratio: float = 0.0

    def __post_init__(self):
        assert self.feat_dir is not None, "Feature directory is required"
        self.feat_dir = Path(self.feat_dir)
        assert self.split is not None, "Split is required"
        assert self.nshard is not None, "Number of shards is required"
        assert self.km_path is not None, "K-means path is required"
        self.km_path = Path(self.km_path)
        assert self.n_clusters is not None, "Number of clusters is required"
        assert self.percent <= 1.0, "Percentage should be less than or equal to 1.0"


@dataclass
class DumpLabelConfig(ProcessConfig):
    feat_dir: str = None
    split: str = None
    km_path: str = None
    nshard: int = None
    rank: int = None
    lab_dir: str = None

    def __post_init__(self):
        assert self.feat_dir is not None, "Feature directory is required"
        self.feat_dir = Path(self.feat_dir)
        assert self.split is not None, "Split is required"
        assert self.km_path is not None, "K-means path is required"
        self.km_path = Path(self.km_path)
        assert self.nshard is not None, "Number of shards is required"
        assert self.rank is not None, "Rank is required"
        assert self.lab_dir is not None, "Label directory is required"
        self.lab_dir = Path(self.lab_dir)
        self.lab_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class CountClustersConfig(ProcessConfig):
    split: str = None
    nshard: int = None
    lab_dir: str = None
    output_dir: str = None

    def __post_init__(self):
        assert self.split is not None, "Split is required"
        assert self.nshard is not None, "Number of shards is required"
        assert self.lab_dir is not None, "Label directory is required"
        self.lab_dir = Path(self.lab_dir)
        assert self.output_dir is not None, "Output directory is required"
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class UploadConfig:
    path: str = None    # Path to the file or directory to process
    log_path: str = None    # Path to the log file
    overwrite: bool = False     # Overwrite existing files
    reverse: bool = False   # Reverse the order of the files
    repo_id: str = None     # HuggingFace repository ID
    dir_in_repo: str = "."      # Path to upload in the repository
    repo_type: str = "dataset"      # Type of the repository
    every_minutes: float = 1    # Upload every F minutes
    delete_after_upload: bool = False   # Delete the file or directory after uploading
    zip: bool = False   # Zip the directory before uploading

    def __post_init__(self):
        assert glob(self.path), f"Path {self.path} does not exist."
        self.path = Path(self.path)
        if self.log_path is not None:
            self.log_path = Path(self.log_path)
        assert self.repo_id is not None, "Repository ID is required"
        self.dir_in_repo = Path(self.dir_in_repo)
        assert self.repo_type in ["dataset", "model"], \
            "Only Dataset and Model repositories are supported"
        assert self.every_minutes > 0, "Upload interval should be positive"


@dataclass
class ModelConfig:
    pretrained: str = "DEFAULT"

    label_rate: int = 25
    sample_rate: int = 25
    input_modality: str = "video"
    extractor_mode: str = "default"
    encoder_layers: int = 24
    encoder_embed_dim: int = 1024
    encoder_ffn_embed_dim: int = 4096
    encoder_attention_heads: int = 16
    activation_fn: str = "gelu"
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.1
    encoder_layerdrop: float = 0.0
    dropout_input: float = 0.0
    dropout_features: float = 0.0
    final_dim: int = 256
    untie_final_proj: bool = False
    layer_norm_first: bool = False
    conv_feature_layers: str = "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2"
    conv_bias: bool = False
    logit_temp: float = 0.1
    target_glu: bool = False
    feature_grad_mult: float = 1.0
    mask_length_audio: int = 10
    mask_prob_audio: float = 0.65
    mask_length_image: int = 10
    mask_prob_image: float = 0.65
    mask_selection: str = "static"
    mask_other: float = 0.0
    no_mask_overlap: bool = False
    mask_min_space: int = 1
    mask_channel_length: int = 64
    mask_channel_prob: float = 0.5
    mask_channel_selection: str = "static"
    mask_channel_other: float = 0.0
    no_mask_channel_overlap: bool = False
    mask_channel_min_space: int = 1
    conv_pos: int = 128
    conv_pos_groups: int = 16
    latent_temp: tuple = field(default_factory=lambda: (2.0, 0.5, 0.999995))
    skip_masked: bool = False
    skip_nomask: bool = False
    resnet_relu_type: str = "prelu"
    resnet_weights: str = None
    sim_type: str = "cosine"
    sub_encoder_layers: int = 0
    audio_feat_dim: int = 104
    modality_dropout: float = 0.0
    audio_dropout: float = 0.0
    modality_fuse: str = "concat"
    selection_type: str = "same_other_seq"
    masking_type: str = "input"
    decoder_embed_dim: int = 2560
    decoder_ffn_embed_dim: int = 3072
    decoder_layers: int = 6
    decoder_layerdrop: float = 0.0
    decoder_attention_heads: int = 4
    decoder_learned_pos: bool = False
    decoder_normalize_before: bool = False
    no_token_positional_embeddings: bool = False
    decoder_dropout: float = 0.1
    decoder_attention_dropout: float = 0.1
    decoder_activation_dropout: float = 0.0
    max_target_positions: int = 2048
    share_decoder_input_output_embed: bool = False
    no_scale_embedding: bool = True
    num_classes: int = 2004
    feature_ds_rate: int = 1


@dataclass
class InferenceConfig:
    source: str = None
    output_dir: str = "demo"
    assistant_model: str = None
    device: str = "cpu"
    cache_dir: str = "models/huggingface"
    visualize: bool = False

    def __post_init__(self):
        self.source = Path(self.source)
        assert self.source.exists() and str(self.source).endswith(EXTENSIONS), \
            f"Only video sources are supported for now (got {self.source})"
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
