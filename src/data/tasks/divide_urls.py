import os
import logging
import pandas as pd
import pyarrow as pa

from pyarrow import parquet as pq
from typing import List
from huggingface_hub import HfFileSystem

from src.data.utils.file_system import prepare_dir

fs = HfFileSystem()

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)

def divide_urls(
        metadata: List[dict],
        volume: int, 
        redivide: bool = False
    ):
    logger.info(msg="Dividing urls into channels")
    """
    Devide list of urls into batch of url.
    """
    metadata_dir = os.path.join(os.getcwd(),'src','data','databases','metadata')
    prepare_dir(metadata_dir,overwrite=redivide)
    for batch_idx, i in enumerate(range(0,len(metadata),volume)):
        df = pd.DataFrame(metadata[i:i+volume])
        channel = "batch_%.5d" % batch_idx
        channels = [channel] * len(df)
        df['channel'] = channels
        path_file = os.path.join(metadata_dir,channel + '.parquet')
        table = df.to_parquet(path=path_file)
    table: pa.Table = pq.read_table("src/data/databases/metadata/batch_00000.parquet")
    with fs.open('datasets/GSU24AI03-SU24AI21/tracked-url-video/metadata/batch_00000.parquet',mode='wb') as f:
        pq.write_table(table,f)
        


        

