import joblib
import numpy as np
from pathlib import Path
from loguru import logger
from configs import LearnKMeansConfig
from sklearn.cluster import MiniBatchKMeans


def load_feature_shard(
    feat_dir: Path,
    split: str,
    nshard: int,
    rank: int,
    percent: float,
) -> np.ndarray:
    feat_file = feat_dir / f"{split}_{rank}_{nshard}.npy"
    len_file = feat_file.with_suffix(".len")
    with open(len_file, "r") as f:
        lengs = [int(line.rstrip()) for line in f]
        offsets = [0] + np.cumsum(lengs[:-1]).tolist()

    if percent < 0:
        return np.load(feat_file, mmap_mode="r")

    nsample = int(np.ceil(len(lengs) * percent))
    indices = np.random.choice(len(lengs), nsample, replace=False)
    feat = np.load(feat_file, mmap_mode="r")
    sampled_feat = np.concatenate(
        [feat[offsets[i]: offsets[i] + lengs[i]] for i in indices], axis=0
    )
    logger.info(
        (
            f"Sampled {nsample} utterances, {len(sampled_feat)} frames "
            f"from shard {rank}/{nshard}"
        )
    )
    return sampled_feat


def learn_kmeans(config: LearnKMeansConfig) -> None:
    np.random.seed(config.seed)

    feat = np.concatenate(
        [
            load_feature_shard(
                feat_dir=config.feat_dir,
                split=config.split,
                nshard=config.nshard,
                rank=r,
                percent=config.percent,
            )
            for r in range(config.nshard)
        ],
        axis=0,
    )
    logger.info(f"loaded feature with dimension {feat.shape}")

    km_model = MiniBatchKMeans(
        n_clusters=config.n_clusters,
        init=config.init,
        max_iter=config.max_iter,
        batch_size=config.batch_size,
        verbose=1,
        compute_labels=False,
        tol=config.tol,
        max_no_improvement=config.max_no_improvement,
        init_size=None,
        n_init=config.n_init,
        reassignment_ratio=config.reassignment_ratio,
    )
    logger.info("K-means model initialized")

    km_model.fit(feat)
    inertia = -km_model.score(feat) / len(feat)
    logger.info("Total intertia: %.2f", inertia)

    joblib.dump(km_model, config.km_path)
    logger.info(f"K-means model saved to {config.km_path}")

    logger.info("Finished successfully")
