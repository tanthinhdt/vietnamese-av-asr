import os
import subprocess
import json
import copy

from src.data.processors.processor import Processor
from src.data.utils import get_logger

class YoutTubeDownloader(Processor):
    """This class used to download video from YouTube."""

    _config_file = 'src/data/databases/command_configs/ytdlp_download.conf'
    _command_download_temp = [
        'yt-dlp',
        '-o',
        'out_path',
        '--config-locations',
        'config_path',
        'video_url',
    ]
    _command_meta_temp = [
        'yt-dlp',
        '--skip-download',
        '--dump-json',
        'video_url',
    ]

    def process(
        self,
        sample: dict,
        video_output_dir: str,
        log_path: str = None,
        *args,
        **kwargs,
    ) -> dict:
        """
        Download video from YouTube.

        sample:
            Dict contains metadata for downloading video.
        video_output_dir:
            Directory will contain video.
        log_path:
            Path of log file.
        return:
            Metadata of processed sample.
        """
        print()
        logger = get_logger(
            name=__name__,
            log_path=log_path,
            is_stream=False,
        )
        logger_ = get_logger(
            log_path=log_path,
            is_stream=False,
            format='%(message)s',
        )

        channel  = sample['channel'][0]
        video_id = sample['video_id'][0]

        logger_.info('-'*35 + f"Yt-downloader processing video id '{video_id}'" + '-'*35)
        video_path = os.path.join(video_output_dir,f"video@{channel}@{video_id}.mp4")

        logger.debug("Load metadata")
        command_meta        = copy.copy(self._command_meta_temp)
        command_meta[-1]    = sample['url'][0]
        metadata = dict()
        metadata = json.loads(
            subprocess.run(
                command_meta,
                shell=False,
                capture_output=True,
            ).stdout.decode('utf-8').strip('\n')
        )
        command_download        = copy.copy(self._command_download_temp)
        command_download[2]     = video_path
        command_download[-2]    = self._config_file
        command_download[-1]    = sample['url'][0]

        logger.debug("Download video")
        subprocess.run(command_download, shell=False, capture_output=False, stdout=None)

        output_sample = copy.copy(sample)
        for k in sample.keys():
            if k not in ('id','channel'):
                output_sample.pop(k)

        output_sample["id"]             = [None]
        output_sample["channel"]        = sample["channel"]
        output_sample["video_id"]       = sample['video_id']
        output_sample["video_name"]     = [os.path.basename(os.path.splitext(video_path)[0])]
        output_sample['demo']           = sample['demo']
        output_sample["uploader"]       = [metadata.get('uploader_id','@no_uploader')[1:]]

        if os.path.isfile(video_path) and os.path.splitext(video_path)[-1] == '.mp4':
            output_sample['id'] = sample['id']

        logger_.info('*'*50 + "Yt-downloader done." + '*'*50)
        return output_sample