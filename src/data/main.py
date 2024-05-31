import os
import sys
sys.path.append(os.getcwd())
import time
import pandas as pd

from pyarrow import parquet as pq
from huggingface_hub import HfFileSystem
from datasets import get_dataset_config_names

from tasks.track_urls import *
from tasks.divide_urls import divide_urls

fs = HfFileSystem()

_PLAYLIST_URLS = [
    "https://www.youtube.com/playlist?list=PLWrhnsc6Cvcrp7HmEWu8q0p95pRyGmpHi",
    "https://www.youtube.com/playlist?list=PL1Vi4Nt_Cpb5oCuSpvekBr9SKkQZkH9TQ",
]


@filter_metadata
def base_duration(meta: dict, threshold: int = 4000):
    return meta['duration'] > 3000 and meta['duration'] < 6000

@filter_metadata
def base_title(meta: dict):
    return "BÍ MẬT ĐỒNG TIỀN" in meta['title'] or "Tự do tài chính" in meta['title']

# metadatas = []
# [metadatas.extend(get_metadatas(url=url,start=1,end=-1)) for url in _PLAYLIST_URLS]
# metadatas = base_duration(metadatas=metadatas)
# divide_urls(metadatas,volume=5,redivide=True)

# df = pd.DataFrame({'video_id': ["5tTiD5rtVwY"],'channel':['batch_99999']},)
# df.to_parquet(path="src/data/databases/metadata/batch_99999.parquet")
print(pq.read_table('/Users/minhnguyen/home/vietnamese-av-asr/data/processed/detected-speaker-clip/metadata/batch_99999.parquet'))
