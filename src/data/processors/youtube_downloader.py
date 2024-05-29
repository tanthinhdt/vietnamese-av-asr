import os
import sys
sys.path.append(os.getcwd())
import subprocess
import json

from copy import copy

from src.data.processors.processor import Processor

class YoutTubeDownloader(Processor):
    
    _config_file = 'src/data/command_configs/ytdlp_download.conf'
    _command_download_temp = [
        'yt-dlp',
        '-o',
        'out_path',
        '--config-locations',
        'config_path',
        'video_id',
    ]
    _command_meta_temp = [
        'yt-dlp',
        '--skip-download',
        '--dump-json',
        'video_id',
    ]

    def process(
        self,
        sample: dict, 
        video_output_dir: str,
        *args,
        **kwargs,
    ) -> dict:  
        command_meta = copy(self._command_meta_temp)
        command_meta[-1] = sample['url'][0]
        metadata = json.loads(
            subprocess.run(
                command_meta,
                shell=False,
                capture_output=True,
            ).stdout.decode('utf-8').strip('\n')
        )
        channel = sample['channel'][0]
        video_path = os.path.join(video_output_dir,f"video@{channel}@{sample['id'][0]}.%(ext)s")
        command_download = copy(self._command_download_temp)
        command_download[2] = video_path
        command_download[-2] = self._config_file
        command_download[-1] = sample['url'][0]
        subprocess.run(command_download, shell=False, capture_output=False, stdout=None)

        output_sample = {
            "id": [None],
            "channel": sample['channel'],
            "uploader": [metadata['uploader_id'][1:]],
            "video_id": [metadata['id']],
            "file_name": [os.path.basename(os.path.splitext(video_path)[0])],
            "duration": [metadata['duration']],
            "fps": [metadata['fps']],
            "asr": [metadata['asr']],    
        }
        if os.path.isfile(video_path) and os.path.splitext(video_path)[-1] == 'mp4':
            output_sample['id'] = sample['id']

        return output_sample
    