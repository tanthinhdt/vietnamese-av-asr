import math
import os
import fairseq
import joblib
import torch
import tqdm
import numpy as np
import torch.nn.functional as F

from npy_append_array import NpyAppendArray
from python_speech_features import logfbank
from scipy.io import wavfile
from src.models.utils import get_logger

logger = get_logger('Clustering', is_stream=True)


class HubertFeatureReader(torch.nn.Module):
    def __init__(self, ckpt_path, layer, max_chunk=1600000, custom_utils=None):
        super().__init__()
        (
            model,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model[0].eval().to(device=self.device)
        self.task = task
        self.layer = layer
        self.max_chunk = max_chunk
        self.stack_order_audio = self.task.cfg.stack_order_audio
        image_crop_size, image_mean, image_std = self.task.cfg.image_crop_size, self.task.cfg.image_mean, self.task.cfg.image_std
        self.transform = custom_utils.Compose([
            custom_utils.Normalize(0.0, 255.0),
            custom_utils.CenterCrop((image_crop_size, image_crop_size)),
            custom_utils.Normalize(image_mean, image_std)])

        self.custom_utils = custom_utils

    def load_feature(self, mix_name, modalities):
        def stacker(feats, stack_order):
            feat_dim = feats.shape[1]
            if len(feats) % stack_order != 0:
                res = stack_order - len(feats) % stack_order
                res = np.zeros([res, feat_dim]).astype(feats.dtype)
                feats = np.concatenate([feats, res], axis=0)
            feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order*feat_dim)
            return feats

        video_fn, audio_fn = mix_name
        video_feats = self.load_image(video_fn)
        audio_fn = audio_fn.split(':')[0]

        sample_rate, wav_data = wavfile.read(audio_fn)
        assert sample_rate == 16_000 and len(wav_data.shape) == 1
        audio_feats = logfbank(wav_data, samplerate=sample_rate).astype(np.float32)
        audio_feats = stacker(audio_feats, self.stack_order_audio)

        if 'visual' not in modalities:
            shape = (audio_feats.shape[0],) + video_feats.shape[1:]
            video_feats = np.zeros(shape)
        if 'audio' not in modalities:
            shape = (video_feats.shape[0],) + audio_feats.shape[1:]
            audio_feats = np.zeros(shape)

        diff = len(audio_feats) - len(video_feats)
        if diff < 0:
            audio_feats = np.concatenate([audio_feats, np.zeros([-diff, audio_feats.shape[-1]], dtype=audio_feats.dtype)])
        elif diff > 0:
            audio_feats = audio_feats[:-diff]

        return video_feats, audio_feats

    def load_image(self, video_fn):
        feats = self.custom_utils.load_video(video_fn)
        feats = self.transform(feats)
        feats = np.expand_dims(feats, axis=-1)
        return feats

    def get_feats(self, path, modalities: list):
        video_feats, audio_feats = self.load_feature(path, modalities)
        with torch.no_grad():
            audio_feats = torch.from_numpy(audio_feats.astype(np.float32)).to(device=self.device)
            video_feats = torch.from_numpy(video_feats.astype(np.float32)).to(device=self.device)
            if self.task.cfg.normalize:
                audio_feats = F.layer_norm(audio_feats, audio_feats.shape[1:])
            video_feats = video_feats.unsqueeze(dim=0).permute((0, 4, 1, 2, 3)).contiguous()
            audio_feats = audio_feats.unsqueeze(dim=0).transpose(1, 2)

            source = {'audio': audio_feats, 'video': video_feats}
            if self.layer == 0:
                ret_conv, output_layer = True, None
            else:
                ret_conv, output_layer = False, self.layer
            feat, _ = self.model.extract_features(
                source=source,
                padding_mask=None,
                mask=False,
                output_layer=output_layer,
                ret_conv=ret_conv
            )
            return feat.squeeze(dim=0)


class ApplyKmeans(object):
    def __init__(self, km_path):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            dist = (
                x.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(x, self.C)
                + self.Cnorm
            )
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = (
                (x ** 2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)


def get_path_iterator(tsv, nshard, rank):
    with open(tsv, "r") as f:
        root = f.readline().rstrip()
        lines = [line.rstrip() for line in f]
        tot = len(lines)
        shard_size = math.ceil(tot / nshard)
        start, end = rank * shard_size, min((rank + 1) * shard_size, tot)
        assert start < end, f"start={start}, end={end}"
        lines = lines[start:end]
        def iterate():
            for line in lines:
                items = line.strip().split("\t")
                yield (items[1], items[2]+':'+items[0]), int(items[3])

        return iterate, len(lines)


def dump_feature(
        tsv_dir, split, extractor, nshard, rank, feat_dir, modalities: list = None, **kwargs
):
    if modalities is None:
        logger.critical("Select modalities to dump feature")
        raise RuntimeError()
    if not {'visual', 'audio'}.intersection(set(modalities)):
        logger.critical("No visual or audio modalities.")
        raise RuntimeError()
    tsv_path = os.path.join(tsv_dir, f"{split}.tsv")
    generator, num = get_path_iterator(tsv_path, nshard, rank)
    iterator = generator()

    feat_path = f"{feat_dir}/{split}.npy"
    leng_path = f"{feat_dir}/{split}.len"

    os.makedirs(feat_dir, exist_ok=True)
    if os.path.exists(feat_path):
        os.remove(feat_path)

    feat_f = NpyAppendArray(feat_path)
    with open(leng_path, "w") as leng_f:
        for path, nsample in tqdm.tqdm(iterator, total=num):
            feat = extractor.get_feats(path, modalities=modalities)
            feat_f.append(feat.cpu().to(torch.float).numpy())
            leng_f.write(f"{len(feat)}\n")
    logger.info("Extract feature finished successfully")


def get_feat_iterator(feat_dir, split):
    feat_path = f"{feat_dir}/{split}.npy"
    leng_path = f"{feat_dir}/{split}.len"
    with open(leng_path, "r") as f:
        lengs = [int(line.rstrip()) for line in f]
        offsets = [0] + np.cumsum(lengs[:-1]).tolist()

    def iterate():
        feat = np.load(feat_path, mmap_mode="r")
        assert feat.shape[0] == (offsets[-1] + lengs[-1])
        for offset, leng in zip(offsets, lengs):
            yield feat[offset: offset + leng]

    return iterate, len(lengs)


def dump_label(feat_dir, split, km_path, lab_dir):
    apply_kmeans = ApplyKmeans(km_path)
    generator, num = get_feat_iterator(feat_dir, split)
    iterator = generator()

    lab_path = f"{lab_dir}/{split}.km"
    os.makedirs(lab_dir, exist_ok=True)
    with open(lab_path, "w") as f:
        for feat in tqdm.tqdm(iterator, total=num):
            lab = apply_kmeans(feat).tolist()
            f.write(" ".join(map(str, lab)) + "\n")
    logger.info("Assign units finished successfully")


def cluster_count():
    unit_pth = "src/models/dataset/vsr/vi/test.km"
    units = open(unit_pth).readlines()
    count_list = []
    for unit_line in units:
        unit_line = unit_line.strip().split(' ')
        int_unit_line = [int(x) for x in unit_line]
        current_count = 1
        counts = []
        for i in range(1, len(int_unit_line)):
            if int_unit_line[i] == int_unit_line[i - 1]:
                current_count += 1
            else:
                counts.append(current_count)
                current_count = 1
        counts.append(current_count)
        str_counts = [str(x) for x in counts]
        count_list.append(' '.join(str_counts) + '\n')
    cluster_counts_pth = unit_pth.replace('.km','.cluster_counts')
    with open(cluster_counts_pth, 'w') as f:
        f.write(''.join(count_list))
    logger.info("Cluster count finished successfully")