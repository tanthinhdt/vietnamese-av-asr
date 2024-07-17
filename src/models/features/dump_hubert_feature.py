import logging
import math
import os
import sys
import fairseq
import torch
import tqdm
import numpy as np
import torch.nn.functional as F

from npy_append_array import NpyAppendArray
from python_speech_features import logfbank
from scipy.io import wavfile

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_hubert_feature")


class HubertFeatureReader(object):
    def __init__(self, ckpt_path, layer, max_chunk=1600000, custom_utils=None):
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
        logger.info(f"TASK CONFIG:\n{self.task.cfg}")
        logger.info(f" max_chunk = {self.max_chunk}")
        logger.info(f"Transform: {self.transform}")

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
            video_feats *= 0
        if 'audio' not in modalities:
            audio_feats *= 0

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


def get_path_iterator(tsv, nshard, rank):
    with open(tsv, "r") as f:
        root = f.readline().rstrip()
        lines = [line.rstrip() for line in f]
        tot = len(lines)
        shard_size = math.ceil(tot / nshard)
        start, end = rank * shard_size, min((rank + 1) * shard_size, tot)
        assert start < end, f"start={start}, end={end}"
        logger.info(
            f"rank {rank} of {nshard}, process {end-start} "
            f"({start}-{end}) out of {tot}"
        )

        lines = lines[start:end]

        def iterate():
            for line in lines:
                items = line.strip().split("\t")
                yield (items[1], items[2]+':'+items[0]), int(items[3])

        return iterate, len(lines)


def dump_feature(
        tsv_dir, split, ckpt_path, layer, nshard, rank, feat_dir, max_chunk, custom_utils=None, modalities: list = None, **kwargs
):
    if modalities is None:
        logger.critical("Select modalities to dump feature")
        exit(1)
    if not {'visual', 'audio'}.intersection(set(modalities)):
        logger.critical("No visual or audio modalities.")
        exit(1)
    reader = HubertFeatureReader(ckpt_path, layer, max_chunk, custom_utils=custom_utils)
    tsv_path = os.path.join(tsv_dir, f"{split}.tsv")
    generator, num = get_path_iterator(tsv_path, nshard, rank)
    iterator = generator()

    feat_path = f"{feat_dir}/{split}_{rank}_{nshard}.npy"
    leng_path = f"{feat_dir}/{split}_{rank}_{nshard}.len"

    os.makedirs(feat_dir, exist_ok=True)
    if os.path.exists(feat_path):
        os.remove(feat_path)

    feat_f = NpyAppendArray(feat_path)
    with open(leng_path, "w") as leng_f:
        for path, nsample in tqdm.tqdm(iterator, total=num):
            feat = reader.get_feats(path, modalities=modalities)
            feat_f.append(feat.cpu().to(torch.float).numpy())
            leng_f.write(f"{len(feat)}\n")
    logger.info("finished successfully")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("tsv_dir")
    parser.add_argument("split")
    parser.add_argument("ckpt_path")
    parser.add_argument("layer", type=int)
    parser.add_argument("nshard", type=int)
    parser.add_argument("rank", type=int)
    parser.add_argument("feat_dir")
    parser.add_argument("--max_chunk", type=int, default=1600000)
    parser.add_argument("--user_dir", type=str, default=None)
    parser.add_argument('--modalities', nargs='+', type=str, required=False, default=None)

    args = parser.parse_args()
    sys.path.append(args.user_dir)
    logger.info(args)
    from src.models.utils import vsp_llm as custom_utils
    kwargs = vars(args)
    kwargs.update({'custom_utils': custom_utils})
    dump_feature(**kwargs)