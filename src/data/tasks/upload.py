import os
import sys

sys.path.append(os.getcwd())

import argparse
from tqdm import tqdm
from logging import getLogger
from huggingface_hub import HfApi
from src.data.utils import zip_dir


logger = getLogger()


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--token",
        type=str,
        default="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",
        required=False,
        help="HuggingFace access token.",
    )
    argparser.add_argument(
        "--src",
        type=str,
        default="/home/khanhphd/Documents/data/phdkhanh2507/testVLR/save",
        required=False,
        help="Path to file or directory to upload.",
    )
    argparser.add_argument(
        "--dest",
        type=str,
        default="vietnamese-speaker-video/video",
        required=False,
        help="Path in repo.",
    )
    argparser.add_argument(
        "--channel-names-path",
        type=str,
        default="phdkhanh2507/testVLR/channel.txt",
        help="Path to file containing channel names to upload.",
    )
    argparser.add_argument(
        "--zip",
        action='store_true',
        default=False,
        help="Whether to automatically zip the directory.",
    )
    argparser.add_argument(
        "--overwrite",
        action='store_true',
        default=False,
        help="Whether to overwrite existing zip files.",
    )
    argparser.add_argument(
        "--clean-up",
        action='store_true',
        default=False,
        help="Whether to clean up zip files after uploading.",
    )
    return argparser.parse_args()


def main(args: argparse.Namespace):
    """
    Main function.
    """
    with open(args.channel_names_path, "r") as f:
        channel_names = f.read().splitlines()

    print("#" * 50 + " Uploading " + "#" * 50)
    api = HfApi()
    for channel_name in tqdm(
        channel_names,
        desc="Processing channels",
        total=len(channel_names),
        unit="channel"
    ):
        print("-" * 50 + f" Processing {channel_name} " + "-" * 50)

        src_path = os.path.join(args.src, channel_name)
        if not os.path.basename(args.src).startswith("stage"):
            if not os.path.exists(src_path):
                print(f"Channel {channel_name} does not exist.")
                continue
            if args.zip:
                zip_dir(src_path, overwrite=args.overwrite)
                src_path = src_path + ".zip"
        else:
            src_path = src_path + ".json"
            if not os.path.exists(src_path):
                print(f"Channel {channel_name} does not exist.")
                continue

        if os.path.isdir(src_path):
            for file_name in os.listdir(src_path):
                file_path = os.path.join(src_path, file_name)
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=os.path.join(args.dest, channel_name, file_name),
                    repo_id="phdkhanh2507/testVLR",
                    repo_type="dataset",
                    commit_message=f"chore: update {args.dest} directory",
                    commit_description=f"Add {channel_name}",
                    token=args.token,
                )
        else:
            api.upload_file(
                path_or_fileobj=src_path,
                path_in_repo=os.path.join(args.dest, channel_name + ".zip"),
                repo_id="phdkhanh2507/testVLR",
                repo_type="dataset",
                commit_message=f"chore: update {args.dest} directory",
                commit_description=f"Add {channel_name}",
                token=args.token,
            )

        if args.clean_up and src_path.endswith("zip"):
            os.remove(src_path)
        print("-" * (13 + len(channel_name) + 2 * 20))

if __name__ == "__main__":
    main(parse_args())
