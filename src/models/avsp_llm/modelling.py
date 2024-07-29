import torch
import logging
import contextlib
import numpy as np
import torch.nn as nn
from pathlib import Path
from .resnet import ResNetEncoder
from .encoder import TransformerEncoder
from .configuration import AVHubertConfig, AVSPLLMConfig
from .utils import compute_mask_indices, load_kmeans_model
from typing import Optional, Tuple, List, Dict, Any
from peft import get_peft_model, LoraConfig
from fairseq.modules import GradMultiply, LayerNorm
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import (
    FeatureExtractionMixin,
    PreTrainedModel,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    GenerationConfig,
)


logging.root.setLevel(logging.WARNING)


class AVHubertFeatureExtractor(FeatureExtractionMixin):
    def __init__(self, config: AVHubertConfig = AVHubertConfig(), **kwargs) -> None:
        super().__init__(**kwargs)
        self.audio_feat_dim = config.audio_feat_dim

        self.size = 88
        self.num_frames = 76
        self.num_channels = 1


class AVSPLLMFeatureExtractor(AVHubertFeatureExtractor):
    def __init__(self, config: AVSPLLMConfig = AVSPLLMConfig(), **kwargs) -> None:
        super().__init__(config, **kwargs)


class AVHubertVideoFeatureEncoder(nn.Module):
    def __init__(self, config: AVHubertConfig) -> None:
        super().__init__()
        self.resnet = ResNetEncoder(relu_type=config.resnet_relu_type)
        self.proj = nn.Linear(self.resnet.backend_out, config.encoder_embed_dim)
        self.encoder = (
            TransformerEncoder(config)
            if config.sub_encoder_layers > 0
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet(x)
        x = self.proj(x.transpose(1, 2))
        if self.encoder is not None:
            x = self.encoder(x)[0].transpose(1, 2)
        else:
            x = x.transpose(1, 2)
        return x


class AVHubertAudioFeatureEncoder(nn.Module):
    def __init__(self, config: AVHubertConfig) -> None:
        super().__init__()
        self.proj = nn.Linear(config.audio_feat_dim, config.encoder_embed_dim)
        self.encoder = (
            TransformerEncoder(config)
            if config.sub_encoder_layers > 0
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x.transpose(1, 2))
        if self.encoder is not None:
            x = self.encoder(x)[0].transpose(1, 2)
        else:
            x = x.transpose(1, 2)
        return x


class AVHubertModel(PreTrainedModel):
    config_class = AVHubertConfig

    def __init__(
        self,
        config: AVHubertConfig = AVHubertConfig(),
        dictionaries: List = [None],
    ) -> None:
        super().__init__(config=config)
        label_rate = config.label_rate
        feature_ds_rate = config.feature_ds_rate
        sample_rate = config.sample_rate
        self.feat2tar_ration = label_rate * feature_ds_rate / sample_rate

        self.feature_extractor_video = AVHubertVideoFeatureEncoder(config)
        self.feature_extractor_audio = AVHubertAudioFeatureEncoder(config)

        if config.modality_fuse == "concat":
            self.encoder_embed_dim = config.encoder_embed_dim * 2
        elif config.modality_fuse == "add":
            self.encoder_embed_dim = config.encoder_embed_dim

        self.post_extract_proj = (
            nn.Linear(self.encoder_embed_dim, config.encoder_embed_dim)
            if self.encoder_embed_dim != config.encoder_embed_dim
            else None
        )

        self.dropout_input = nn.Dropout(config.dropout_input)
        self.dropout_features = nn.Dropout(config.dropout_features)

        if self.config.final_dim > 0:
            final_dim = config.final_dim
        else:
            final_dim = config.encoder_embed_dim

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(config.audio_feat_dim).uniform_()
            if config.masking_type == "input"
            else torch.FloatTensor(config.encoder_embed_dim).uniform_()
        )

        self.encoder = TransformerEncoder(self.config)
        self.layer_norm = LayerNorm(self.encoder_embed_dim)

        self.target_glu = None
        if config.target_glu:
            self.target_glu = nn.Sequential(
                nn.Linear(config.final_dim, config.final_dim * 2),
                nn.GLU(),
            )

        if config.untie_final_proj:
            self.final_proj = nn.Linear(
                config.encoder_embed_dim,
                final_dim * len(dictionaries),
            )
        else:
            self.final_proj = nn.Linear(config.encoder_embed_dim, final_dim)

        # modules below are not needed during fine-tuning
        if any([d is None for d in dictionaries]):
            self.num_classes = config.num_classes
        else:
            self.num_classes = sum([len(d) for d in dictionaries])
        self.label_embs_concat = nn.Parameter(
            torch.FloatTensor(self.num_classes, final_dim)
        )
        nn.init.uniform_(self.label_embs_concat)

    def apply_input_mask(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        target_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, T = x.shape[:3]
        is_audio = True if len(x.shape) == 3 else False

        if is_audio:
            mask_prob = self.config.mask_prob_audio
            mask_length = self.config.mask_length_audio
        else:
            mask_prob = self.config.mask_prob_image
            mask_length = self.config.mask_length_image

        if mask_prob > 0:
            mask_indices, starts, ends, batch_indexes = compute_mask_indices(
                (B, T),
                padding_mask,
                mask_prob,
                mask_length,
                self.config.mask_selection,
                self.config.mask_other,
                min_masks=2,
                no_overlap=self.config.no_mask_overlap,
                min_space=self.config.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x = x.transpose(1, 2).contiguous()  # [B, T, C, H, W]
            if B == 1:
                x[mask_indices] = 0
            elif is_audio:
                x[mask_indices] = self.mask_emb
            elif self.config.selection_type == "same_other_seq":
                perm = (torch.arange(B) + torch.randint(low=1, high=B, size=(1,))) % B
                x_perm = x[perm]
                x[mask_indices] = x_perm[mask_indices]
            elif self.config.selection_type == "same_seq":
                batch_indexes_, other_indexes = [], []
                for batch_index, start, end in zip(batch_indexes, starts, ends):
                    length = end - start
                    other_start = np.setdiff1d(
                        np.arange(T), np.arange(max(0, start - length), end)
                    )
                    if len(other_start) > 0:
                        other_start = np.random.choice(other_start, size=1)
                    else:
                        other_start = 0
                    other_end = other_start + length
                    other_indexes.append(
                        np.arange(other_start, other_end).clip(max=T - 1)
                    )
                    batch_indexes_.append(
                        np.zeros([length], dtype=np.int64) + batch_index
                    )
                batch_indexes = np.concatenate(batch_indexes_)
                other_indexes = np.concatenate(other_indexes)
                x[mask_indices] = x[batch_indexes, other_indexes]
            x = x.transpose(1, 2).contiguous()
        else:
            mask_indices = None

        if self.config.mask_channel_prob > 0:
            logging.warn("No mask channel prob for input masking")
        return x, mask_indices

    def apply_feature_mask(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        target_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        assert all((
            self.config.mask_prob_audio == self.config.mask_prob_image,
            self.config.mask_length_audio == self.config.mask_length_image,
        )), "masking prob/length for image/audio be same for feature masking"

        mask_prob = self.config.mask_prob_audio
        mask_length = self.config.mask_length_image
        if mask_prob > 0:
            mask_indices, _, _, _ = compute_mask_indices(
                (B, T),
                padding_mask,
                mask_prob,
                mask_length,
                self.config.mask_selection,
                self.config.mask_other,
                min_masks=2,
                no_overlap=self.config.no_mask_overlap,
                min_space=self.config.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None

        if self.config.mask_channel_prob > 0:
            mask_channel_indices, _, _, _ = compute_mask_indices(
                (B, C),
                None,
                self.config.mask_channel_prob,
                self.config.mask_channel_length,
                self.config.mask_channel_selection,
                self.config.mask_channel_other,
                no_overlap=self.config.no_mask_channel_overlap,
                min_space=self.config.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        return x, mask_indices

    def forward_features(
        self,
        source: Dict[str, torch.Tensor],
        modality: str,
    ) -> torch.Tensor:
        extractor = eval(f"self.feature_extractor_{modality}")
        if self.config.feature_grad_mult > 0:
            features = extractor(source)
            if self.config.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.config.feature_grad_mult)
        else:
            with torch.no_grad():
                features = extractor(source)
        return features

    def forward_targets(
        self,
        features: torch.Tensor,
        mask_indices: torch.Tensor,
        target_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        # Trim features to ensure labels exist and then get aligned labels
        feat_tsz = features.size(2)
        targ_tsz = min([t.size(1) for t in target_list])
        if self.feat2tar_ratio * feat_tsz > targ_tsz:
            feat_tsz = int(targ_tsz / self.feat2tar_ratio)
            features = features[..., :feat_tsz]
            if mask_indices is not None:
                mask_indices = mask_indices[..., :feat_tsz]
        target_inds = torch.arange(feat_tsz).float() * self.feat2tar_ratio
        target_list = [t[:, target_inds.long()] for t in target_list]
        return features, mask_indices, target_list

    def forward_padding_mask(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def compute_logits(self, feats: torch.Tensor, emb_mat: torch.Tensor) -> torch.Tensor:
        # feats: [B, T, F], emb_mat: [V, F]
        if self.config.sim_type == "dot":
            logits = torch.matmul(feats, emb_mat.transpose(0, 1))
        elif self.config.sim_type == "cosine":
            batch_size, timesteps, emb_dim = feats.size()
            feats_ = feats.view(-1, emb_dim)
            # [B*T, V]
            nom = (feats_.unsqueeze(dim=1) * emb_mat.unsqueeze(dim=0)).sum(dim=-1)
            # [B*T, V]
            denom = (
                (feats_**2).sum(dim=-1).sqrt().unsqueeze(dim=1)
                * (emb_mat**2).sum(dim=-1).sqrt().unsqueeze(dim=0)
            )
            logits = (nom / denom.clamp(min=1e-6)).view(batch_size, timesteps, -1)
        else:
            raise NotImplementedError
        logits = logits / self.config.logit_temp
        return logits

    def forward(
        self,
        source: Dict[str, torch.Tensor],
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """output layer is 1-based"""
        src_audio, src_video = source["audio"], source["video"]
        if mask and self.masking_type == "input":
            src_video, mask_indices_video = self.apply_input_mask(
                src_video, padding_mask, target_list
            )
            src_audio, mask_indices_audio = self.apply_input_mask(
                src_audio, padding_mask, target_list
            )
            mask_indices = torch.logical_or(mask_indices_audio, mask_indices_video)
        else:
            src_audio, src_video, mask_indices = src_audio, src_video, None

        # [B, F, T]
        features_audio = self.forward_features(src_audio, modality="audio")
        features_video = self.forward_features(src_video, modality="video")

        if self.training:
            modality_drop_prob, audio_drop_prob = np.random.random(), np.random.random()
            if modality_drop_prob < self.config.modality_dropout:
                if audio_drop_prob < self.config.audio_dropout:
                    features_audio = 0 * features_audio
                else:
                    features_video = 0 * features_video

        if self.config.modality_fuse == "concat":
            features = torch.cat([features_audio, features_video], dim=1)
        elif self.config.modality_fuse == "add":
            features = features_audio + features_video

        if target_list is not None:
            features, mask_indices, target_list = self.forward_targets(
                features, mask_indices, target_list
            )

        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        if self.config.masking_type == "feature" and mask:
            x, mask_indices = self.apply_feature_mask(
                features, padding_mask, target_list
            )
        else:
            x = features

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool
        x, _ = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1,
        )

        if features_only:
            return {"x": x, "padding_mask": padding_mask, "features": features}

        label_embs_list = self.label_embs_concat.split(self.num_classes, 0)
        proj_x = self.final_proj(x)
        if self.config.untie_final_proj:
            proj_x_list = proj_x.chunk(len(self.num_classes), dim=-1)
        else:
            proj_x_list = [proj_x for _ in self.num_classes]

        # [[B*T, V]]
        logit_list = [
            self.compute_logits(proj, emb).view(-1, num_class)
            for proj, emb, num_class in zip(
                proj_x_list, label_embs_list, self.num_classes
            )
        ]

        mask = torch.logical_and(mask_indices, ~padding_mask).view(-1)
        unmask = torch.logical_and(~mask_indices, ~padding_mask).view(-1)  # [B*T]
        logit_m_list = [logit[mask] for logit in logit_list]
        logit_u_list = [logit[unmask] for logit in logit_list]
        target_m_list = [target.view(-1)[mask].long() for target in target_list]
        target_u_list = [target.view(-1)[unmask].long() for target in target_list]

        return {
            "logit_m_list": logit_m_list,
            "logit_u_list": logit_u_list,
            "target_m_list": target_m_list,
            "target_u_list": target_u_list,
            "padding_mask": padding_mask,
            "features_pen": features_pen,
        }

    def extract_features(
        self,
        source: Dict[str, torch.Tensor],
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
        ret_conv: bool = False,
        output_layer: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        res = self.forward(
            source,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,
            output_layer=output_layer,
        )
        feature = res["features"] if ret_conv else res["x"]
        return feature, res["padding_mask"]

    def extract_units(
        self,
        source: Dict[str, torch.Tensor],
        padding_mask: torch.Tensor = None,
        mask: bool = False,
        ret_conv: bool = False,
        output_layer: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        res = self.forward(
            source,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,
            output_layer=None,
        )

        feature = res["features"] if ret_conv else res["x"]
        proj_x = self.final_proj(feature)
        # B T
        units = (
            torch
            .matmul(proj_x, self.label_embs_concat.transpose(0, 1))
            .argmax(dim=-1)
        )
        return units

    def extract_finetune(
        self,
        source: Dict[str, torch.Tensor],
        padding_mask: torch.Tensor = None,
        mask: bool = False,
        ret_conv: bool = False,
        output_layer: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src_audio, src_video = source["audio"], source["video"]
        if mask and self.config.masking_type == "input":
            src_video, _ = self.apply_input_mask(
                src_video, padding_mask, target_list=None
            )
            src_audio, _ = self.apply_input_mask(
                src_audio, padding_mask, target_list=None
            )
        else:
            src_audio, src_video, _ = src_audio, src_video, None

        # features: [B, F, T]
        if src_audio is not None and src_video is None:
            features_audio = self.forward_features(
                src_audio, modality="audio"
            )
            features_video = features_audio.new_zeros(
                features_audio.size(0),
                self.encoder_embed_dim,
                features_audio.size(-1)
            )
        elif src_audio is None and src_video is not None:
            features_video = self.forward_features(src_video, modality="video")
            features_audio = features_video.new_zeros(
                features_video.size(0),
                self.encoder_embed_dim,
                features_video.size(-1)
            )
        elif src_audio is not None and src_video is not None:
            features_video = self.forward_features(src_video, modality="video")
            features_audio = self.forward_features(
                src_audio, modality="audio"
            )

        if self.config.modality_fuse == "concat":
            features = torch.cat([features_audio, features_video], dim=1)
        elif self.config.modality_fuse == "add":
            features = features_audio + features_video

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        unmasked_features = features.clone()

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool
        x, _ = self.encoder(
            features,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1,
        )

        return x, padding_mask

    def get_extra_losses(
        self,
        net_output: Dict[str, torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[str]]:
        extra_losses = []
        names = []
        if "features_pen" in net_output:
            extra_losses.append(net_output["features_pen"])
            names.append("features_pen")

        return extra_losses, names

    def remove_pretraining_modules(self) -> None:
        self.target_glu = None
        self.final_proj = None

    def compute_nce(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        negs: torch.Tensor,
    ) -> torch.Tensor:
        neg_is_pos = (pos == negs).all(-1)
        pos = pos.unsqueeze(0)
        targets = torch.cat([pos, negs], dim=0)

        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1).type_as(x)
        logits /= self.config.logit_temp
        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")
        logits = logits.transpose(0, 1)  # (num_x, num_cls+1)
        return logits


class HubertEncoderWrapper(nn.Module):
    def __init__(
        self,
        config: AVHubertConfig,
        dictionaries: List = [None],
    ) -> None:
        super().__init__()
        self.w2v_model = AVHubertModel(config, dictionaries)

    def forward(
        self,
        source: Dict[str, torch.Tensor],
        padding_mask: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
        }
        x, padding_mask = self.w2v_model.extract_finetune(**w2v_args)
        return {
            "encoder_out": x,  # T x B x C
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask,
        }

    def reorder_encoder_out(
        self,
        encoder_out: Dict[str, torch.Tensor],
        new_order: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out["encoder_out"].index_select(
                1, new_order
            )
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(0, new_order)
        if encoder_out["padding_mask"] is not None:
            encoder_out["padding_mask"] = encoder_out["padding_mask"].index_select(
                0, new_order
            )
        return encoder_out


class AVSPLLMModel(PreTrainedModel):
    config_class = AVSPLLMConfig

    def __init__(
        self,
        config: AVSPLLMConfig = AVSPLLMConfig(),
        dictionaries: List = [None],
    ) -> None:
        super().__init__(config=config)
        current_dir = Path(__file__).resolve().parent
        self.km_path = current_dir / config.km_path
        if not self.km_path.is_file():
            repo_id = self.config._name_or_path
            self.km_path = f"{repo_id}/model.km"
        self.km_path = str(self.km_path)
        self.C, self.Cnorm = load_kmeans_model(self.km_path)

        self.encoder = HubertEncoderWrapper(config, dictionaries)
        self.encoder.w2v_model.remove_pretraining_modules()

        self.avfeat_to_llm = nn.Linear(
            config.encoder_embed_dim, config.decoder_embed_dim
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        decoder_4bit = AutoModelForCausalLM.from_pretrained(
            config.llm_ckpt_path,
            quantization_config=bnb_config,
            cache_dir=config.cache_dir,
            trust_remote_code=True,
        )
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.decoder = get_peft_model(decoder_4bit, lora_config)
        self.decoder.print_trainable_parameters()

    def apply_kmeans(self, feat: torch.Tensor) -> torch.Tensor:
        dist = (
            feat.squeeze(0).pow(2).sum(1, keepdim=True)
            - 2 * torch.matmul(feat.squeeze(0), self.C)
            + self.Cnorm
        )
        cluster_counts = dist.argmin(dim=1)

        current_counts = 1
        counts = []
        for i in range(1, len(cluster_counts)):
            if cluster_counts[i] == cluster_counts[i - 1]:
                current_counts += 1
            else:
                counts.append(current_counts)
                current_counts = 1
        counts.append(current_counts)

        return torch.tensor(counts)

    def deduplicate(
        self,
        feat: torch.Tensor,
        cluster_counts: torch.Tensor,
    ) -> torch.Tensor:
        results_tensor = []
        start_idx = 0
        for clutser_num in cluster_counts:
            end_idx = start_idx + clutser_num
            slice = feat[:, start_idx:end_idx, :]
            mean_tensor = torch.mean(slice, dim=1, keepdim=True)
            results_tensor.append(mean_tensor)
            start_idx = end_idx

        assert cluster_counts.sum().item() == feat.size()[1], \
            f"{cluster_counts.sum().item()} != {feat.size()[1]}"

        return torch.cat(results_tensor, dim=1)

    def embed(
        self,
        source: Dict[str, torch.Tensor],
        padding_mask: torch.Tensor,
        target_list: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        ft = self.config.freeze_finetune_updates <= kwargs.get("num_updates", -1)
        with torch.no_grad() if not ft else contextlib.ExitStack():
            output = self.encoder(source, padding_mask, **kwargs)

        cluster_counts = self.apply_kmeans(output["encoder_out"])

        output["encoder_out"] = self.avfeat_to_llm(output["encoder_out"])

        reduced_enc_out = self.deduplicate(output["encoder_out"], cluster_counts)
        reduced_enc_out = reduced_enc_out.to(self.decoder.device)
        B, T, D = reduced_enc_out.size()

        instruction = source["text"]
        instruction_embedding = self.decoder.model.model.embed_tokens(instruction)

        llm_input = torch.cat((instruction_embedding, reduced_enc_out), dim=1)

        if target_list is None:
            return llm_input, None

        labels = target_list.clone()
        labels_embedding = self.decoder.model.model.embed_tokens(labels)

        llm_input = torch.cat((llm_input, labels_embedding), dim=1)

        llm_labels = labels.clone()
        llm_labels[llm_labels == 0] = -100

        _, instruction_embedding_t, _ = instruction_embedding.size()
        target_ids = (
            torch.full((B, T + instruction_embedding_t), -100).long().to(labels.device)
        )
        llm_labels = torch.cat((target_ids, llm_labels), dim=1)

        return llm_input, llm_labels

    def forward(
        self,
        source: Dict[str, torch.Tensor],
        padding_mask: torch.Tensor,
        target_list: torch.Tensor = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        llm_input, llm_labels = self.embed(
            source, padding_mask, target_list, **kwargs
        )
        return self.decoder(
            inputs_embeds=llm_input.to(torch.float16), labels=llm_labels, return_dict=True
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[Dict[str, torch.Tensor]] = None,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> Any:
        llm_input, _ = self.embed(**inputs, **kwargs)
        self.decoder.config.use_cache = True
        return self.decoder.generate(
            inputs_embeds=llm_input,
            **generation_config,
            **kwargs,
        )
