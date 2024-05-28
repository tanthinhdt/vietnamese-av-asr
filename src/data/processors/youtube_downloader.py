# import os
# import sys
# sys.path.append(os.getcwd())
# import subprocess

# from src.data.processors.processor import Processor

# class YoutTubeDownloader(Processor):
    
#     _config_file = 'src/data/command_configs/ytdlp_download.conf'
#     _command_ = [
#         'yt-dlp',
#         '-o',
#         None,
#         '--config-locations',
#         None,
#         None,
#     ]
#     def process(
#         self,
#         sample: dict, 
#         video_output_dir: str,
#         *args,
#         **kwargs,
#     ) -> dict:
        
#         channel = sample['channel']
#         url = sample['url']
#         video_path = os.path.join(video_output_dir,channel,f"video@{channel}@%(id)s.%(ext)s")
#         self._command_[2] = video_path
#         self._command_[-2] = self._config_file
#         self._command_[-1] = url
#         subprocess.call(self._command_, shell=True, stdout=None)
#         if os.path.isfile(video_path):
#             video_id = os.path.basename(os.path.splitext(video_path)[0])
#             ext = os.path.splitext(video_path)[1]
#             file_name = f"video@{channel}@{video_id}.{ext}"
#         output_sample = {
#             "channel": channel,
#             "video_id": video_id,
#             "file_name": file_name
#         }

#         return output_sample