import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from datasets import Dataset
from configs import CreateManifestConfig


def filter(sample: dict, split: str, data_dir: Path) -> bool:
    if sample["split"] != split:
        return False
    id = sample["id"]
    shard = sample["shard"]

    visual_file = data_dir / f"visual/{shard}/{id}.mp4"
    if not visual_file.exists():
        logger.error(f"File {visual_file} does not exist.")
        return False

    audio_file = data_dir / f"audio/{shard}/{id}.wav"
    if not audio_file.exists():
        logger.error(f"File {audio_file} does not exist.")
        return False

    return True


def create_manifest(config: CreateManifestConfig) -> None:
    mapping_df = pd.read_json(
        os.path.join(config.data_dir, "mapping.json"),
        dtype={
            "id": "string",
            "shard": "string",
        },
    )

    df = pd.read_parquet(config.metadata_path)
    logger.info(f"Found {len(df)} ids.")

    df = pd.merge(
        df,
        mapping_df,
        how="left",
        on=["id", "shard", "split"],
    )
    df = (
        Dataset.from_pandas(df)
        .filter(
            filter,
            fn_kwargs={"split": config.split, "data_dir": config.data_dir},
            num_proc=10,
            load_from_cache_file=False,
        )
        .to_pandas()
    )
    logger.info(f"Get {len(df)} after filtering")

    df = df.groupby("channel", group_keys=False).apply(
        lambda x: x.sample(frac=config.frac)
    )
    logger.info(f"Stratified sampling {len(df)} samples")

    manifest = []
    texts = []
    progress_bar = tqdm(enumerate(df.itertuples()), total=len(df))
    for i, sample in progress_bar:
        rel_visual_path = os.path.join(
            "visual", sample.shard, f"{sample.id}.mp4",
        )
        rel_audio_path = os.path.join(
            "audio", sample.shard, f"{sample.id}.wav",
        )

        manifest.append(
            "\t".join(
                [
                    f"{sample.id}-{config.src_lang}-{config.dst_lang}",
                    rel_visual_path,
                    rel_audio_path,
                    str(sample.video_num_frames),
                    str(sample.audio_num_frames),
                ]
            )
        )
        texts.append(sample.transcript)

    with open(config.output_dir / f"{config.split}.tsv", "w") as f:
        f.write(str(config.data_dir) + "\n")
        f.write("\n".join(manifest) + "\n")
    with open(config.output_dir / f"{config.split}.wrd", "w") as f:
        f.write("\n".join(texts) + "\n")

    logger.info(f"{config.split} set have {len(texts)} sample")
