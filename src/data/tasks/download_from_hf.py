import os
import sys

sys.path.append(os.getcwd())

import zipfile
import argparse
from tqdm import tqdm
from logging import getLogger
from huggingface_hub import hf_hub_download


logger = getLogger()


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--token",
        type=str,
        required=True,
        help="HuggingFace access token.",
    )
    argparser.add_argument(
        "--subfolder",
        type=str,
        required=True,
        help="Subfolder in HuggingFace repo.",
    )
    argparser.add_argument(
        "--channel-names-path",
        type=str,
        help="Path to file containing channel names to upload.",
    )
    argparser.add_argument(
        "--dest",
        type=str,
        required=True,
        help="Path to destination directory.",
    )
    return argparser.parse_args()


def main(args: argparse.Namespace):
    """
    Main function.
    """
    save_dir = os.path.join(args.dest, args.subfolder)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(args.channel_names_path, "r") as f:
        channel_names = set(f.read().splitlines())
    channel_names -= set(os.listdir(save_dir))

    for channel_name in tqdm(
        list(channel_names),
        desc="Downloading channels",
        total=len(channel_names),
        unit="channel"
    ):
        print("#" * 50 + f" Downloading and extracting {channel_name} " + "#" * 50)
        file_name = channel_name
        file_name += ".json" if os.path.basename(args.subfolder).startswith("stage") else ".zip"
        hf_hub_download(
            repo_id="fptu/vlr",
            filename=file_name,
            subfolder=args.subfolder,
            repo_type="dataset",
            local_dir=args.dest,
            local_dir_use_symlinks=False,
            token=args.token,
        )
        file_path = os.path.join(args.dest, args.subfolder, file_name)
        if file_name.endswith(".zip"):
            with zipfile.ZipFile(file_path) as zf:
                zf.extractall(save_dir)
            os.remove(file_path)


if __name__ == "__main__":
    main(parse_args())
