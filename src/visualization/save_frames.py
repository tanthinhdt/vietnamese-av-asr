import cv2
import shutil
from tqdm import tqdm
from glob import glob
from pathlib import Path
from loguru import logger
from configs import SaveFramesConfig


def save_frames(config: SaveFramesConfig):
    video_files = [Path(video_file) for video_file in glob(str(config.video_file))]
    logger.info(f"Found {len(video_files)} video files")

    for video_file in tqdm(video_files):
        logger.info(f"Processing {video_file}")
        output_dir = config.output_dir / video_file.with_suffix("").name

        if not config.overwrite and output_dir.exists():
            logger.info(f"Skipping {video_file} as it already exists")
            continue

        video = cv2.VideoCapture(str(video_file))
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        frame_count = 0
        while True:
            success, frame = video.read()
            if not success:
                break
            frame_count += 1
            frame_path = output_dir / f"{frame_count}.jpg"
            cv2.imwrite(frame_path, frame)

        logger.info(f"Saved {frame_count} frames to {output_dir}")
        video.release()

    logger.info("Done!")
