import copy
import logging
import os
import subprocess

from src.models.taskers.tasker import Tasker
from src.models.taskers.checker import Checker
from src.models.utils.media import get_duration


class Normalizer(Tasker):

    FPS: int = 25
    SR: int = 16_000

    AV1_TO_H264_CMD = [
        'ffmpeg', '-y',
        '-loglevel', 'panic',
        '-c:v', 'libaom-av1',
        '-i', 'video_path',
        '-c:v', 'libx264',
        '-c:a', 'pcm_s16le',
        '-ss', '0',
        '-t', 'duration',
        '-f', 'avi',
        'output_path',
    ]

    ANY_TO_H264_PCM_CMD = [
        'ffmpeg', '-y',
        '-loglevel', 'panic',
        '-i', 'video_path',
        '-map', '0:v:0',
        '-map', '0:a:0',
        '-c:v', 'libx264',
        '-c:a', 'pcm_s16le',
        '-ss', '0',
        '-t', 'duration',
        '-f', 'avi',
        'output_path',
    ]

    ADD_BLANK_SOUND_CMD = [
        'ffmpeg', '-y',
        '-loglevel', 'panic',
        '-f', 'lavfi',
        '-i', 'anullsrc=channel_layout=mono',
        '-i', 'video_path',
        '-r', '%d' % FPS,
        '-ar', '%d' % SR,
        '-c:v', 'copy',
        '-c:a', 'pcm_s16le',
        '-ss', '0',
        '-t', 'duration',
        '-f', 'avi',
        'output_path'
    ]

    ADD_BACKGROUND_VISUAL_CMD = [
        'ffmpeg', '-y',
        '-loglevel', 'panic',
        '-f', 'lavfi',
        '-i', 'color=c=black',
        '-i', 'video_path',
        '-c:v', 'libx264',
        '-c:a', 'pcm_s16le',
        '-ss', '0',
        '-t', 'duration',
        '-r', '%d' % FPS,
        '-pix_fmt', 'yuv420p',
        '-vf', 'scale=400:400',
        '-f', 'avi',
        'output_path',
    ]

    def __init__(self, checker: Checker = None):
        super().__init__()

    def do(self, metadata_dict: dict, checker: Checker=Checker()):
        if metadata_dict['has_v']:
            if metadata_dict['v_codec'] == 'av1':
                metadata_dict['video_path'] = self._normalize(
                    cmd_=self.AV1_TO_H264_CMD,
                    meta_dict=metadata_dict,
                    output_name='nv',
                    fail_msg="Convert vcodec 'av1' -> 'libx264' fail."
                )
            elif metadata_dict['v_codec'] != 'h264':
                metadata_dict['video_path'] = self._normalize(
                    cmd_=self.ANY_TO_H264_PCM_CMD,
                    meta_dict=metadata_dict,
                    output_name='nv',
                    fail_msg="Convert vcodec to 'libx264' fail."
                )

            if not metadata_dict['has_a']:
                metadata_dict['video_path'] = self._normalize(
                    cmd_=self.ADD_BLANK_SOUND_CMD,
                    meta_dict=metadata_dict,
                    output_name='na',
                    fail_msg="Add blank sound fail."
                )
        else:
            metadata_dict['video_path'] = self._normalize(
                cmd_=self.ADD_BACKGROUND_VISUAL_CMD,
                meta_dict=metadata_dict,
                output_name='nv',
                fail_msg="Add background visual fail."
            )

        tmp = checker.do(video_path=metadata_dict['video_path'])
        tmp['result_video_path'] = metadata_dict.get('result_video_path', metadata_dict['video_path'])

        return tmp

    def _normalize(self, cmd_: list, meta_dict: dict, output_name: str,  fail_msg: str = None) -> str:
        cmd = copy.copy(cmd_)
        output_path = os.path.join(
            'data', 'external', f'{output_name}.mp4'
        )

        cmd[cmd.index('video_path')] = meta_dict['video_path']
        cmd[-1] = output_path
        if 'duration' in cmd:
            cmd[cmd.index('duration')] = str(meta_dict['duration'])

        subprocess.run(cmd, shell=False, capture_output=False, stdout=None)

        if not os.path.isfile(output_path):
            logging.warning(fail_msg)
            exit([1])
        return output_path