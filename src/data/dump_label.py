import tqdm
import torch
import joblib
import numpy as np
from pathlib import Path
from typing import Union
from loguru import logger
from configs import DumpLabelConfig


class ApplyKmeans:
    def __init__(self, km_path: Path) -> None:
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            dist = (
                x.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(x, self.C)
                + self.Cnorm
            )
            return dist.argmin(dim=1).cpu().numpy()

        dist = (
            (x ** 2).sum(1, keepdims=True)
            - 2 * np.matmul(x, self.C_np)
            + self.Cnorm_np
        )
        return np.argmin(dist, axis=1)


def get_feat_iterator(feat_dir: Path, split: str, nshard: int, rank: int) -> tuple:
    feat_file = feat_dir / f"{split}_{rank}_{nshard}.npy"
    leng_file = feat_file.with_suffix(".len")
    with open(leng_file, "r") as f:
        lengs = [int(line.rstrip()) for line in f]
        offsets = [0] + np.cumsum(lengs[:-1]).tolist()

    def iterate():
        feat = np.load(feat_file, mmap_mode="r")
        assert feat.shape[0] == (offsets[-1] + lengs[-1])
        for offset, leng in zip(offsets, lengs):
            yield feat[offset: offset + leng]

    return iterate, len(lengs)


def dump_label(config: DumpLabelConfig) -> None:
    apply_kmeans = ApplyKmeans(config.km_path)
    logger.info(f"K-means model loaded from {config.km_path}")

    generator, num = get_feat_iterator(
        feat_dir=config.feat_dir,
        split=config.split,
        nshard=config.nshard,
        rank=config.rank,
    )
    iterator = generator()
    logger.info("Feature iterator created")

    lab_file = config.lab_dir / f"{config.split}_{config.rank}_{config.nshard}.km"
    with open(lab_file, "w") as f:
        for feat in tqdm.tqdm(iterator, total=num):
            # feat = torch.from_numpy(feat).cuda()
            lab = apply_kmeans(feat).tolist()
            f.write(" ".join(map(str, lab)) + "\n")
    logger.info(f"Labels dumped to {lab_file}")
    logger.info("Finished successfully")
