import subprocess


def get_duration(file: str):
    _duration_cmd = "ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 %s" % file

    duration = float(subprocess.run(_duration_cmd, shell=True, capture_output=True, stdout=None).stdout)

    return duration


def get_fps(video_file: str):
    _visual_fps_cmd = "ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of default=noprint_wrappers=1:nokey=1 %s" % video_file

    fps = int(subprocess.run(_visual_fps_cmd, shell=True, capture_output=True, stdout=None).stdout.decode().split('/')[0])

    return fps


def get_sr(audio_file: str):
    _audio_fps_cmd = "ffprobe -v error -select_streams a:0 -show_entries stream=sample_rate -of default=noprint_wrappers=1:nokey=1 %s" % audio_file

    sr = int(subprocess.run(_audio_fps_cmd, shell=True, capture_output=True, stdout=None).stdout)

    return sr


