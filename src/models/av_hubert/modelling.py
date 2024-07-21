import torch
import logging
import numpy as np
import torch.nn as nn
from .resnet import ResNetEncoder
from .configuration import AVHubertConfig
from typing import Optional, Tuple, List, Dict
from transformers import FeatureExtractionMixin, PreTrainedModel
from fairseq.modules import GradMultiply, LayerNorm
from fairseq.models.wav2vec.wav2vec2 import TransformerEncoder
from fairseq.modules import GradMultiply, LayerNorm


class AVHubertFeatureExtractor(FeatureExtractionMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class AVHubertVideoFeatureEncoder(nn.Module):
    def __init__(self, config: AVHubertConfig) -> None:
        super().__init__()
        self.resnet = ResNetEncoder(relu_type=config.resnet_relu_type)
        self.proj = nn.Linear(config.audio_feat_dim, self.resnet.backend_out)
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


def compute_mask_indices(
    shape: Tuple[int, int],
    padding_mask: Optional[torch.Tensor],
    mask_prob: float,
    mask_length: int,
    mask_type: str = "static",
    mask_other: float = 0.0,
    min_masks: int = 0,
    no_overlap: bool = False,
    min_space: int = 0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape
    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        padding_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_type: how to compute mask lengths
            static = fixed size
            uniform = sample from uniform distribution [mask_other, mask_length*2]
            normal = sample from normal distribution with mean mask_length and stdev mask_other. mask is min 1 element
            poisson = sample from possion distribution with lambda = mask length
        min_masks: minimum number of masked spans
        no_overlap: if false, will switch to an alternative recursive algorithm that prevents spans from overlapping
        min_space: only used if no_overlap is True, this is how many elements to keep unmasked between spans
    """
    def find_runs(x):
        """Find runs of consecutive items in an array."""

        # ensure array
        x = np.asanyarray(x)
        if x.ndim != 1:
            raise ValueError("only 1D array supported")
        n = x.shape[0]

        # handle empty array
        if n == 0:
            return np.array([]), np.array([]), np.array([])

        else:
            # find run starts
            loc_run_start = np.empty(n, dtype=bool)
            loc_run_start[0] = True
            np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
            run_starts = np.nonzero(loc_run_start)[0]

            # find run values
            run_values = x[loc_run_start]

            # find run lengths
            run_lengths = np.diff(np.append(run_starts, n))

            return run_values, run_starts, run_lengths

    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    all_num_mask = int(
        # add a random number for probabilistic rounding
        mask_prob * all_sz / float(mask_length)
        + np.random.rand()
    )

    all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    for i in range(bsz):
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            num_mask = int(
                # add a random number for probabilistic rounding
                mask_prob * sz / float(mask_length)
                + np.random.rand()
            )
            num_mask = max(min_masks, num_mask)
        else:
            sz = all_sz
            num_mask = all_num_mask

        if mask_type == "static":
            lengths = np.full(num_mask, mask_length)
        elif mask_type == "uniform":
            lengths = np.random.randint(mask_other, mask_length * 2 + 1, size=num_mask)
        elif mask_type == "normal":
            lengths = np.random.normal(mask_length, mask_other, size=num_mask)
            lengths = [max(1, int(round(x))) for x in lengths]
        elif mask_type == "poisson":
            lengths = np.random.poisson(mask_length, size=num_mask)
            lengths = [int(round(x)) for x in lengths]
        else:
            raise Exception("unknown mask selection " + mask_type)

        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)

        if no_overlap:
            mask_idc = []

            def arrange(s, e, length, keep_length):
                span_start = np.random.randint(s, e - length)
                mask_idc.extend(span_start + i for i in range(length))

                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - keep_length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts

            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = np.fromiter(
                    (e - s if e - s >= length + min_space else 0 for s, e in parts),
                    np.int,
                )
                l_sum = np.sum(lens)
                if l_sum == 0:
                    break
                probs = lens / np.sum(lens)
                c = np.random.choice(len(parts), p=probs)
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = np.asarray(mask_idc)
        else:
            min_len = min(lengths)
            if sz - min_len <= num_mask:
                min_len = sz - num_mask - 1

            mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)

            mask_idc = np.asarray(
                [
                    mask_idc[j] + offset
                    for j in range(len(mask_idc))
                    for offset in range(lengths[j])
                ]
            )

        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))

    min_len = min([len(m) for m in mask_idcs])
    batch_indexes, starts, ends = [], [], []
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        mask[i, mask_idc] = True
        vals, run_starts, run_lengths = find_runs(mask[i])
        start_indices, lengths = run_starts[vals == True], run_lengths[vals == True]
        starts.append(start_indices)
        ends.append(start_indices + lengths)
        batch_indexes.append(np.zeros([len(start_indices)]) + i)
    return (
        mask,
        np.concatenate(starts).astype(np.int64),
        np.concatenate(ends).astype(np.int64),
        np.concatenate(batch_indexes).astype(np.int64),
    )


class AVHubertModel(PreTrainedModel):
    config_class = AVHubertConfig

    def __init__(
        self,
        config: AVHubertConfig = AVHubertConfig(),
        dictionaries: dict = None,
    ) -> None:
        super().__init__(config=config)
        label_rate = config.label_rate
        feature_ds_rate = config.feature_ds_rate
        sample_rate = config.sample_rate
        self.feat2tar_ration = label_rate * feature_ds_rate / sample_rate

        self.feature_extractor_video = AVHubertVideoFeatureEncoder(config)
        self.feature_extractor_audio = AVHubertAudioFeatureEncoder(config)

        if config.modality_fuse == "concat":
            self.encoder_embed_dim = config.feat_proj_dim * 2
        elif config.modality_fuse == "add":
            self.encoder_embed_dim = config.feat_proj_dim

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
            if self.masking_type == "input"
            else torch.FloatTensor(config.encoder_embed_dim).uniform_()
        )

        self.encoder = TransformerEncoder(self.config)
        self.layer_norm = LayerNorm(config.embed)

        self.target_glu = None
        if self.config.target_glu:
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
            logging.info("cannot find dictionary. assume will be used for fine-tuning")
        else:
            self.num_classes = [len(d) for d in dictionaries]
            self.label_embs_concat = nn.Parameter(
                torch.FloatTensor(sum(self.num_classes), final_dim)
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
            logging.info("No mask channel prob for input masking")
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
