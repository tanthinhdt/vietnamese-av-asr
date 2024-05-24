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
    '''
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
    '''
    with zipfile.ZipFile(zip_path['file'], 'r') as z:
        z.extractall(output_dir)
        logger.info(f'Extracted {zip_path} to {output_dir}')
    if delete_after_extract:
        os.remove(zip_path['file'])
        logger.info(f'Deleted {zip_path}')


def zip_dir(
    path: str,
    zip_path: str,
    logger: Logger,
    overwrite: bool = False,
) -> None:
    '''
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
    '''
    if not os.path.exists(zip_path) or overwrite:
        shutil.make_archive(zip_path, 'zip', path)
        logger.info(f'Zipped {path} to {zip_path}')
    else:
        logger.warning(f'{zip_path} already exists, skipping zipping')
