from typing import Tuple, Optional
from dataclasses import dataclass, field
from transformers import PretrainedConfig


class AVHubertConfig(PretrainedConfig):
    model_type = "av_hubert"

    def __init__(
        self,
        label_rate: int = 25,
        sample_rate: int = 25,
        input_modality: str = "video",
        extractor_mode: str = "default",
        encoder_layers: int = 24,
        encoder_embed_dim: int = 1024,
        encoder_ffn_embed_dim: int = 4096,
        encoder_attention_heads: int = 16,
        activation_fn: str = "gelu",
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        encoder_layerdrop: float = 0.0,
        dropout_input: float = 0.0,
        dropout_features: float = 0.0,
        final_dim: int = 256,
        untie_final_proj: bool = False,
        layer_norm_first: bool = False,
        conv_feature_layers: str = "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
        conv_bias: bool = False,
        logit_temp: float = 0.1,
        target_glu: bool = False,
        feature_grad_mult: float = 1.0,
        mask_length_audio: int = 10,
        mask_prob_audio: float = 0.65,
        mask_length_image: int = 10,
        mask_prob_image: float = 0.65,
        mask_selection: str = "static",
        mask_other: float = 0.0,
        no_mask_overlap: bool = False,
        mask_min_space: int = 1,
        mask_channel_length: int = 64,
        mask_channel_prob: float = 0.5,
        mask_channel_selection: str = "static",
        mask_channel_other: float = 0.0,
        no_mask_channel_overlap: bool = False,
        mask_channel_min_space: int = 1,
        conv_pos: int = 128,
        conv_pos_groups: int = 16,
        latent_temp: Tuple[float, float, float] = (2.0, 0.5, 0.999995),
        skip_masked: bool = False,
        skip_nomask: bool = False,
        resnet_relu_type: str = "prelu",
        resnet_weights: str = None,
        sim_type: str = "cosine",
        sub_encoder_layers: int = 0,
        audio_feat_dim: int = 104,
        modality_dropout: float = 0.0,
        audio_dropout: float = 0.0,
        modality_fuse: str = "concat",
        selection_type: str = "same_other_seq",
        masking_type: str = "input",
        decoder_embed_dim: int = 2560,
        decoder_ffn_embed_dim: int = 3072,
        decoder_layers: int = 6,
        decoder_layerdrop: float = 0.0,
        decoder_attention_heads: int = 4,
        decoder_learned_pos: bool = False,
        decoder_normalize_before: bool = False,
        no_token_positional_embeddings: bool = False,
        decoder_dropout: float = 0.1,
        decoder_attention_dropout: float = 0.1,
        decoder_activation_dropout: float = 0.0,
        max_target_positions: int = 2048,
        share_decoder_input_output_embed: bool = False,
        no_scale_embedding: bool = True,
        num_classes: int = 2004,
        feature_ds_rate: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.label_rate = label_rate
        self.sample_rate = sample_rate
        self.input_modality = input_modality
        self.extractor_mode = extractor_mode
        self.encoder_layers = encoder_layers
        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_ffn_embed_dim = encoder_ffn_embed_dim
        self.encoder_attention_heads = encoder_attention_heads
        self.activation_fn = activation_fn
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.encoder_layerdrop = encoder_layerdrop
        self.dropout_input = dropout_input
        self.dropout_features = dropout_features
        self.final_dim = final_dim
        self.untie_final_proj = untie_final_proj
        self.layer_norm_first = layer_norm_first
        self.conv_feature_layers = conv_feature_layers
        self.conv_bias = conv_bias
        self.logit_temp = logit_temp
        self.target_glu = target_glu
        self.feature_grad_mult = feature_grad_mult
        self.mask_length_audio = mask_length_audio
        self.mask_prob_audio = mask_prob_audio
        self.mask_length_image = mask_length_image
        self.mask_prob_image = mask_prob_image
        self.mask_selection = mask_selection
        self.mask_other = mask_other
        self.no_mask_overlap = no_mask_overlap
        self.mask_min_space = mask_min_space
        self.mask_channel_length = mask_channel_length
        self.mask_channel_prob = mask_channel_prob
        self.mask_channel_selection = mask_channel_selection
        self.mask_channel_other = mask_channel_other
        self.no_mask_channel_overlap = no_mask_channel_overlap
        self.mask_channel_min_space = mask_channel_min_space
        self.conv_pos = conv_pos
        self.conv_pos_groups = conv_pos_groups
        self.latent_temp = latent_temp
        self.skip_masked = skip_masked
        self.skip_nomask = skip_nomask
        self.resnet_relu_type = resnet_relu_type
        self.resnet_weights = resnet_weights
        self.sim_type = sim_type
        self.sub_encoder_layers = sub_encoder_layers
        self.audio_feat_dim = audio_feat_dim
        self.modality_dropout = modality_dropout
        self.audio_dropout = audio_dropout
        self.modality_fuse = modality_fuse
        self.selection_type = selection_type
        self.masking_type = masking_type
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_ffn_embed_dim = decoder_ffn_embed_dim
        self.decoder_layers = decoder_layers
        self.decoder_layerdrop = decoder_layerdrop
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_learned_pos = decoder_learned_pos
        self.decoder_normalize_before = decoder_normalize_before
        self.no_token_positional_embeddings = no_token_positional_embeddings
        self.decoder_dropout = decoder_dropout
        self.decoder_attention_dropout = decoder_attention_dropout
        self.decoder_activation_dropout = decoder_activation_dropout
        self.max_target_positions = max_target_positions
        self.share_decoder_input_output_embed = share_decoder_input_output_embed
        self.no_scale_embedding = no_scale_embedding
        self.num_classes = num_classes
        self.feature_ds_rate = feature_ds_rate


class AVSPLLMConfig(AVHubertConfig):
    model_type = "avsp_llm"

    def __init__(
        self,
        llm_ckpt_path: str = "vilm/vinallama-2.7b",
        cache_dir: str = "models/huggingface",
        no_pretrained_weights: bool = False,
        final_dropout: float = 0.1,
        apply_mask: bool = False,
        mask_length: int = 10,
        mask_prob: float = 0.5,
        masking_updates: int = 0,
        layerdrop: float = 0.0,
        normalize: bool = False,
        data: str = None,
        w2v_args: dict = None,
        freeze_finetune_updates: int = 0,
        km_path: str = "model.km",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.llm_ckpt_path = llm_ckpt_path
        self.cache_dir = cache_dir
        self.no_pretrained_weights = no_pretrained_weights
        self.final_dropout = final_dropout
        self.apply_mask = apply_mask
        self.mask_length = mask_length
        self.mask_prob = mask_prob
        self.masking_updates = masking_updates
        self.layerdrop = layerdrop
        self.normalize = normalize
        self.data = data
        self.w2v_args = w2v_args
        self.freeze_finetune_updates = freeze_finetune_updates
        self.km_path = km_path


@dataclass
class GenerationConfig:
    """
    For more details, please visit:
    https://huggingface.co/docs/transformers/main_classes/text_generation
    """
    max_length: int = field(default=20)
    max_new_tokens: Optional[int] = field(default=None)
    min_length: int = field(default=0)
    min_new_tokens: Optional[int] = field(default=None)
    max_time: Optional[float] = field(default=None)

    do_sample: bool = field(default=False)
    num_beams: int = field(default=1)
    num_beam_groups: int = field(default=1)

    temperature: float = field(default=1.0)
    top_k: int = field(default=50)
    top_p: float = field(default=1.0)
    min_p: Optional[float] = field(default=None)
    typical_p: float = field(default=1.0)
    epsilon_cutoff: float = field(default=0.0)
    eta_cutoff: float = field(default=0.0)
    diversity_penalty: float = field(default=0.0)
    repetition_penalty: float = field(default=1.0)
    encoder_repetition_penalty: float = field(default=1.0)
    length_penalty: float = field(default=1.0)
    no_repeat_ngram_size: float = field(default=0)
    exponential_decay_length_penalty: Optional[Tuple[int, float]] = field(default=None)
