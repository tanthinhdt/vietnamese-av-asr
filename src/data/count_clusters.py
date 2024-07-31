from tqdm import tqdm
from loguru import logger
from configs import CountClustersConfig


def count_clusters(config: CountClustersConfig) -> None:
    samples = []
    num_frames_before = 0
    for rank in range(config.nshard):
        label_file = config.lab_dir / f"{config.split}_{rank}_{config.nshard}.km"
        with open(label_file) as f:
            for line in f.readlines():
                samples.append(line)
                num_frames_before += len(line.split())
    logger.info(f"Loaded {len(samples)} samples")
    logger.info(f"Total number of frames: {num_frames_before}")

    label_file = config.output_dir / f"{config.split}.km"
    with open(label_file, 'w') as f:
        f.write('\n'.join(samples))
    logger.info(f"Saved labels to {label_file}")

    counts_list = []
    num_frames_after = 0
    for unit_line in tqdm(samples):
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
        counts_list.append(' '.join(str_counts) + '\n')
        num_frames_after += len(counts)
    logger.info(f"Total number of frames after dedup will be: {num_frames_after}")
    logger.info(f"Total number of frames removed: {num_frames_before - num_frames_after}")

    cluster_counts_file = config.output_dir / f"{config.split}.cluster_counts"
    with open(cluster_counts_file, 'w') as f:
        f.write(''.join(counts_list))
    logger.info(f"Saved cluster counts to {cluster_counts_file}")

    logger.info("Done!")
