import os
import shutil
import zipfile
from logging import Logger


def extract_zip(
    zip_path: str,
    output_dir: str,
    logger: Logger,
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
    logger : Logger
        Logger
    delete_after_extract : bool, optional
        Delete the zip file after extracting, by default False
    """
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(output_dir)
        logger.info(f"Extracted {zip_path} to {output_dir}")
    if delete_after_extract:
        os.remove(zip_path)
        logger.info(f"Deleted {zip_path}")


def zip_dir(
    dir_path: str,
    output_dir: str,
    logger: Logger,
    overwrite: bool = False,
    delete_after_zip: bool = False,
) -> None:
    """
    Zip a directory.

    Parameters
    ----------
    path : str
        Path to the directory to zip
    zip_path : str
        Path to save the zip file
    logger : Logger
        Logger
    overwrite : bool, optional
        Overwrite the existing zip file, by default False
    """
    output_path = os.path.join(output_dir, os.path.basename(dir_path) + ".zip")
    if not os.path.exists(output_path) or overwrite:
        shutil.make_archive(
            base_name=output_path[:-4],
            format="zip",
            root_dir=os.path.dirname(dir_path),
            base_dir=os.path.basename(dir_path),
        )
        logger.info(f"Zipped {dir_path} to {output_path}")
    else:
        logger.warning(f"{output_path} already exists, skipping zipping")
    if delete_after_zip:
        shutil.rmtree(dir_path)
        logger.info(f"Deleted {dir_path}")
