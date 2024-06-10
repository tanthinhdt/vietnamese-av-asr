import os
import sys

sys.path.append(os.getcwd())

import cv2
import shutil
import logging
import argparse
import numpy as np
import pandas as pd
import mediapipe as mp
import moviepy.editor as mpe
from glob import glob
from datasets import Dataset
from src.utils import get_default_arg_parser, get_logger


class Cropper:
    """
    This class is used to crop mouth region.
    """
    def __init__(self) -> None:
        self.landmark_detector = mp.solutions.face_mesh.FaceMesh()
        self.mouth_landmark_idxes = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        ]

    def process(
        self, sample: dict,
        visual_output_dir: str,
        size: int = 96,
    ) -> dict:
        """
        """
        visual_output_path = os.path.join(visual_output_dir, sample["id"] + ".mp4")

        if not os.path.exists(visual_output_path):
            mouths = []
            num_frames = 0

            for frame in mpe.VideoFileClip(sample["visual"]).iter_frames():
                num_frames += 1
                mouth = self.crop_mouth(frame, size)
                if mouth is None or mouth.shape[0] == 0 or mouth.shape[1] == 0:
                    continue
                mouths.append(mouth)

            if len(mouths) == num_frames:
                self.write_video(
                    video_path=visual_output_path,
                    frames=mouths,
                    size=size,
                    fps=sample["fps"],
                )
            else:
                sample["id"] = None

        return sample

    def crop_mouth(self, frame: np.ndarray, size: int) -> np.ndarray:
        """
        """
        face_landmarks = self.landmark_detector.process(frame).multi_face_landmarks

        if face_landmarks:
            mouth_landmarks = np.array([
                [landmark.x, landmark.y] for landmark in face_landmarks[0].landmark
            ])[self.mouth_landmark_idxes, :]
            center_x = np.mean(mouth_landmarks[:, 0]) * frame.shape[1]
            min_x = int(center_x - size / 2)
            max_x = int(center_x + size / 2)
            center_y = np.mean(mouth_landmarks[:, 1]) * frame.shape[0]
            min_y = int(center_y - size / 2)
            max_y = int(center_y + size / 2)
            return frame[min_y:max_y, min_x:max_x]

        return None

    def write_video(
        self, video_path: str,
        frames: list,
        size: int,
        fps: int,
    ) -> None:
        """
        """
        mpe.VideoFileClip.write_videofile(
            mpe.ImageSequenceClip(
                [cv2.resize(frame, (size, size)) for frame in frames],
                fps=fps,
            ),
            video_path,
            logger=None,
        )


def get_args() -> argparse.Namespace:
    parser = get_default_arg_parser(
        description="Crop mouth region in video",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        required=True,
        help="Path to metadata file",
    )
    parser.add_argument(
        "--visual-dir",
        type=str,
        required=True,
        help="Path to visual directory",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=96,
        help="Size of cropped mouth region",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=None,
        help="Size of cropped mouth region",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to output directory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Reverse order of files",
    )
    return parser.parse_args()


def main(args: argparse.Namespace, logger: logging.Logger) -> None:
    logger.info("Cropping mouth region in video")

    cropper = Cropper()
    metadata_paths = sorted(glob(args.metadata_path), reverse=args.reverse)
    metadata_dst_dir = os.path.join(args.output_dir, "metadata")
    os.makedirs(metadata_dst_dir, exist_ok=True)

    for i, metadata_path in enumerate(metadata_paths):
        channel = os.path.basename(metadata_path)[:-8]
        logger.info(f"[{i + 1}/{len(metadata_paths)}] Processing {channel}")

        visual_src_dir = os.path.join(args.visual_dir, channel)
        visual_dst_dir = os.path.join(args.output_dir, "visual", channel)
        os.makedirs(visual_dst_dir, exist_ok=True)

        metadata_dst_path = os.path.join(metadata_dst_dir, os.path.basename(metadata_path))
        if os.path.exists(metadata_dst_path) and not args.overwrite:
            logger.info(f"Skipping {metadata_path}")
            if os.path.exists(visual_src_dir):
                shutil.rmtree(visual_src_dir, ignore_errors=True)
                logger.info(f"Removed {visual_src_dir}")
            continue

        metadata = pd.read_parquet(metadata_path)
        metadata["visual"] = metadata["id"].apply(
            lambda x: os.path.join(visual_src_dir, x + ".mp4")
        )
        metadata = Dataset.from_pandas(metadata)
        logger.info(f"Loaded {len(metadata)} samples")

        metadata = metadata.map(
            lambda sample: cropper.process(
                sample=sample,
                visual_output_dir=visual_dst_dir,
                size=args.size,
            ),
            load_from_cache_file=False,
            remove_columns=["visual"],
        )
        metadata = metadata.filter(
            lambda sample: sample["id"] is not None,
            num_proc=args.num_proc,
            load_from_cache_file=False,
        )
        logger.info(f"Processed {len(metadata)} samples")

        metadata.to_parquet(metadata_dst_path)
        logger.info(f"Saved metadata to {metadata_dst_path}")

        shutil.rmtree(visual_src_dir, ignore_errors=True)
        logger.info(f"Removed {visual_src_dir}")

    logger.info("Cropping mouth region in video completed")


if __name__ == "__main__":
    args = get_args()
    logger = get_logger(name="crop_mouth")
    main(args=args, logger=logger)
