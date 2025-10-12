import os
import zipfile

from huggingface_hub import HfFileSystem
import argparse as arg


REPO_ID = "tanthinhdt/vasr"
VASR_DIR = "data/raw/vasr/"


def get_args() -> arg.Namespace:
    parser = arg.ArgumentParser()
    parser.add_argument(
        'no_shard',
        default=None,
        action='store',
        nargs='+',
    )

    return parser.parse_args()

def download_and_extract_shard(
        parser: arg.Namespace
):
    hf = HfFileSystem()
    for shard in parser.no_shard:
        _shard = f"{shard:0>3}"
        r_vpath, l_vpath = (os.path.join("datasets", REPO_ID, "visual", f"{_shard}.zip"),
                            os.path.join(VASR_DIR, "visual", f"{_shard}.zip"))
        r_apath, l_apath = (os.path.join("datasets", REPO_ID, "audio", f"{_shard}.zip"),
                            os.path.join(VASR_DIR, "audio", f"{_shard}.zip"))
        v_dir = os.path.join(VASR_DIR, "visual")
        a_dir = os.path.join(VASR_DIR, "audio")

        if not os.path.exists(l_apath):
            hf.get_file(
                rpath=r_apath,
                lpath=l_apath,
                revision="main",
                recursive=True,
            )

        extract_zip(
            zip_path=l_apath,
            output_dir=a_dir,
            shard=_shard,
            delete_after_extract=False
        )

        if not os.path.exists(l_vpath):
            hf.get_file(
                rpath=r_vpath,
                lpath=l_vpath,
                revision="main",
                recursive=True,
            )

        extract_zip(
            zip_path=l_vpath,
            output_dir=v_dir,
            shard=_shard,
            delete_after_extract=False
        )


def extract_zip(
    zip_path: str,
    output_dir: str,
    shard: str,
    delete_after_extract: bool = False,
) -> None:
    """
    Extract a zip file.

    Parameters
    ----------
    zip_path : str
        Path to the zip file
    output_dir : str
        Path to the output directory
    shard: str

    delete_after_extract : bool, optional
        Delete the zip file after extracting, by default False
    """
    shard_path = os.path.join(output_dir, shard)
    if os.path.isdir(shard_path) and os.listdir(shard_path):
        return
    os.makedirs(shard_path, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(output_dir)
    if delete_after_extract:
        os.remove(zip_path)


if __name__ == "__main__":
    args = get_args()
    download_and_extract_shard(parser=args)
