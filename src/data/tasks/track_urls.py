import logging
import subprocess
import json
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(name=__name__)

__all__ = [
    'get_metadatas',
    'filter_metadata'
]

def get_metadatas(url: str, start: int = 1, end: int = 10):
    logger.info(msg="Tracking urls")
    """
    Get metadata of videos.
    """
    cmd = [
        "yt-dlp",
        "--skip-download",
        "--flat-playlist",
        "--playlist-items",
        f"{start}:{end}:1",
        "--dump-json",
        url,
    ]

    result = subprocess.run(
        args=cmd,
        shell=False,
        capture_output=True,
        stdout=None
    )
    _metas = map(
        lambda s: json.loads(s.strip()),
        result.stdout.decode().strip().split('\n')
    )
    
    short_metas = []
    for meta in _metas:
        short_metas.append({
            'video_id': meta['id'],
            'url': meta['url'],
            'duration': meta['duration'],
            'title': meta['title']
        })

    return short_metas

def filter_metadata(_func: callable):
    def _filter(metadatas: List[dict], *args, **kwargs):
        logger.info(msg="Filtering urls")
        result = type(metadatas)()
        for meta in metadatas:
            try:
                if _func(meta, *args, **kwargs):
                    _meta = {
                        'video_id': meta['video_id'],
                    }
                    result.append(_meta)
            except (ValueError, KeyError, TypeError):
                pass
        return result
    return _filter

