import os
import pandas as pd
import subprocess
import json
import shutil
import datasets

from huggingface_hub import HfFileSystem
from typing import List

from src.data.utils import get_logger, zip_dir

logger = get_logger(name=__name__,is_stream=True)

fs = HfFileSystem()


def get_metadata(url: str) -> List[dict]:
    """
    Get metadata of videos.

    url:
        Url of video or playlist.
    start:
        Starting index of playlist.
    end:
        Ending index of playlist.
    return:
        List of metadata.
    """ 
    if url is not None:
        if os.path.isfile(url):
            with open(url, mode='r') as f:
                _urls = set(f.readlines())
        else:
            _urls = {url}
    
    cmd = [
        "yt-dlp",
        "--skip-download",
        "--flat-playlist",
        "--playlist-items",
        "1::1",
        "--dump-json",
        'url',
    ]
    metadata = []
    for _url in _urls:
        cmd[-1] = _url
        result = subprocess.run(
            args=cmd,
            shell=False,
            capture_output=True,
            stdout=None
        )
        try:
            _metas = map(
                lambda s: json.loads(s.strip()),
                result.stdout.decode().strip().split('\n')
            )
            for meta in _metas:
                metadata.append({
                    'video_id': meta['id'],
                })
        except json.decoder.JSONDecodeError as e:
            logger.info(f"The url '{url}' is not valid.")
            exit(111)
    return metadata


def divide_metadata(
        metadata: List[dict],
        volume: int = 1,
        channel: str = None,
        overwrite: bool = False,
        demo: bool = False
    ) -> None:
    """
    Divide list of urls into batch of url, batch saved file .parquet.

    metadata:
        List of metadata.
    volume:
        Volume for each batch.
    metadata_dir:
        Directory container batch of urls (parquet file).
    channel:
        Indicate name for demo channel.
    demo: 
        Demo or collection
    """
    if channel is not None:
        if os.path.isfile(channel):
            f = open(channel,mode='r')
            channels = set(f.readlines())
            f.close()
        else:
            channels = [channel]
    else:
        channels = []

    available_channels = set(datasets.get_dataset_config_names("GSU24AI03-SU24AI21/tracked-url-video",trust_remote_code=True)) - {'all','batch_88888'}
    n_available_channels = len(available_channels)
    _channels = list(set(channels) - available_channels)

    for batch_idx, i in enumerate(range(0,len(metadata)//volume*volume,volume),start=n_available_channels+10000):
        df = pd.DataFrame(metadata[i:i+volume])
        df['demo'] = [demo] * len(df)
        _channel = "batch%.5d" % batch_idx
        if _channels:
            _channel = _channels.pop(0)
        df['channel'] = [_channel] * len(df)
        path_file = os.path.join('datasets','GSU24AI03-SU24AI21','tracked-url-video@main','metadata',_channel + '.parquet')
        if demo or overwrite or not fs.isfile(path=path_file):
            with fs.open(path_file,'wb') as f:
                df.to_parquet(f)
            logger.info(f"Metadata tracked in '{path_file}'.")
        else:
            logger.info(f"Metadata in '{path_file}' exist.")
            exit(111)


def track_video_file(
    file_path: str,
    channel_name: str,
    demo: bool = True,
) -> None:
    """
    Prepare metadata for file path.

    file_path:
        Video path.
    channel_name:
        Name of channel.
    """
    video_name = f'video@{channel_name}@{"a"*11}'
    prefix,_ = os.path.split(file_path)
    channel_dir = os.path.join(prefix,channel_name)
    video_path = os.path.join(channel_dir,video_name + '.mp4')

    os.makedirs(channel_dir,exist_ok=True)
    shutil.copy(src=file_path,dst=video_path)
    zip_file = zip_dir(zip_dir=channel_dir,overwrite=True)

    sample                  = dict()
    sample['id']            = ['a'*11]
    sample['channel']       = [channel_name]
    sample['video_id']      = [sample['id']]
    sample['video_name']    = [video_name]
    sample['demo']          = [demo]

    with fs.open(f'datasets/GSU24AI03-SU24AI21/downloaded-vietnamese-video@main/metadata/{channel_name}.parquet', mode='wb') as f:
        pd.DataFrame(data=sample).to_parquet(f)
    logger.info('Uploaded video file to \'downloaded\' repo.')

    fs.put(lpath=zip_file,rpath=f'datasets/GSU24AI03-SU24AI21/downloaded-vietnamese-video@main/video/{channel_name}.zip',recursive=False)
    logger.info('Uploaded metadata to \'downloaded\' repo.')

    shutil.rmtree(channel_dir)
    os.remove(path=zip_file)
    
    