import os
import subprocess
import json
import copy

from src.data.processors.processor import Processor


class YoutTubeDownloader(Processor):    
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
        *args,
        **kwargs,
    ) -> dict:  
        channel = sample['channel'][0]
        video_path = os.path.join(video_output_dir,f"video@{channel}@{sample['id'][0]}.mp4")
        try:
            command_meta = copy(self._command_meta_temp)
            command_meta[-1] = sample['url'][0]
            metadata = dict()
            metadata = json.loads(
                subprocess.run(
                    command_meta,
                    shell=False,
                    capture_output=True,
                ).stdout.decode('utf-8').strip('\n')
            )
            command_download = copy(self._command_download_temp)
            command_download[2] = video_path
            command_download[-2] = self._config_file
            command_download[-1] = sample['url'][0]
            subprocess.run(command_download, shell=False, capture_output=False, stdout=None)
        except json.decoder.JSONDecodeError:
            pass
        output_sample = copy.copy(sample)
        for k in sample.keys:
            if k not in ('id','channel'):
                output_sample.pop(k)

        output_sample["id"]                    = [None]
        output_sample["channel"]               = sample["channel"]
        output_sample["video_id"]              = [metadata.get('id','no video id')]
        output_sample["video_name"]            = [os.path.basename(os.path.splitext(video_path)[0])]
        output_sample["duration"]              = [metadata.get('duration',-1)]
        output_sample["visual_fps"]            = [metadata.get('fps',-1)]
        output_sample["audio_fps"]             = [metadata.get('asr',-1)]
        
        if os.path.isfile(video_path) and os.path.splitext(video_path)[-1] == '.mp4':
            output_sample['id'] = sample['id']
        
        return output_sample