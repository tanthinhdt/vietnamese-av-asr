import os
from huggingface_hub import HfFileSystem

_TASK_URL = {
    'track': "GSU24AI03-SU24AI21/tracked-url-video",
    'download': "GSU24AI03-SU24AI21/downloaded-vietnamese-video",
    'asd': "GSU24AI03-SU24AI21/detected-speaker-clip",
    'crop': "GSU24AI03-SU24AI21/cropped-mouth-clip",
    'vndetect': "GSU24AI03-SU24AI21/detected-vietnamese-clip",
    'transcribe': "GSU24AI03-SU24AI21/transcribed-vietnamese-audio",
}

fs = HfFileSystem()


def reset_demo(channel_name: str):
    """
    Reset channel in repository, prepare to demo data collect.

    channel_name:
        Name of channel.
    """
    for task in _TASK_URL:
        metadata_path = os.path.join(
            "datasets", _TASK_URL[task]+"@main", "metadata", f"{channel_name}.parquet"
        )
        if fs.isfile(metadata_path):
            fs.rm(
                path=metadata_path,
                commit_message=f'Delete {channel_name} to prepare demo.',
                commit_description=f'Delete {metadata_path}',
            )


def check_metadata_demo(task: str, channel_name: str) -> bool:
    """
    Check metadata of each task in repos.

    task:
        Name of task.
    channel_name:
        Name of channel.
    return:
         If metadata file exist on repo.
    """
    metadata_path = os.path.join(
        "datasets", _TASK_URL[task]+"@main", "metadata", f"{channel_name}.parquet"
    )
    if fs.isfile(metadata_path):
        return True
    return False
