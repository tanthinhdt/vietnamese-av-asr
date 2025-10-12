from huggingface_hub import HfApi, HfFileSystem
import subprocess

fs = HfFileSystem()
hf = HfApi()

# hf.delete_folder(
#     path_in_repo='src/',
#     repo_id='nguyenminh4099/Demo',
#     revision='main',
#     repo_type='space',
# )
#
hf.upload_folder(
    repo_id='nguyenminh4099/Demo',
    revision='main',
    folder_path='/Users/minhnguyen/home/vietnamese-av-asr/',
    path_in_repo='.',
    repo_type='space',
    ignore_patterns=[
        'README.md',
        'LICENSE',
    ],
)
