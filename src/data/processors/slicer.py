import os
import moviepy.editor as mp
from .processor import Processor


class Slicer(Processor):
    """
    This processor is used to slice video into segments and save them into audio and visual.
    """
    def process(
        self, visual_output_dir: str,
        sample: dict,
        audio_output_dir: str,
        fps: int = 25,
        clip_duration: float = 3.0,
        clip_overlap: float = 1.0,
        keep_last: bool = True,
    ) -> dict:
        """
        Split video into audio and visual.
        :param sample:              Sample.
        :param visual_output_dir:   Path to visual output directory.
        :param audio_output_dir:    Path to audio output directory.
        :param fps:                 FPS of output video.
        :param clip_duration:       Duration of each clip.
        :param clip_overlap:        Overlap between clips.
        :param keep_last:           Keep last clip if it is longer than half of clip duration.
        :return:                    Samples with paths to audio and visual.
        """
        segment_ids = []
        with mp.VideoFileClip(sample["video"][0]) as video:
            duration = video.duration
            sampling_rate = int(video.audio.fps)

            if duration >= clip_duration:
                video = video.set_fps(fps)
                # Split video into segments.
                start = 0
                end = clip_duration
                while end <= duration:
                    segment_id = self.save_visual_and_audio_clip(
                        id=sample["id"][0],
                        video=video,
                        start=start,
                        end=end,
                        visual_dir=visual_output_dir,
                        audio_dir=audio_output_dir,
                    )
                    segment_ids.append(segment_id)

                    start += clip_duration - clip_overlap
                    end = start + clip_duration
                if keep_last and int(duration - start) >= 0.5 * clip_duration:
                    segment_id = self.save_visual_and_audio_clip(
                        id=sample["id"][0],
                        video=video,
                        start=duration - clip_duration,
                        end=duration,
                        visual_dir=visual_output_dir,
                        audio_dir=audio_output_dir,
                    )
                    segment_ids.append(segment_id)
            else:
                segment_ids.append(None)

        return {
            "id": segment_ids,
            "channel": sample["channel"] * len(segment_ids),
            "duration": [clip_duration] * len(segment_ids),
            "fps": [fps] * len(segment_ids),
            "sampling_rate": [sampling_rate] * len(segment_ids),
        }

    def save_visual_and_audio_clip(
        self, id: str,
        video: mp.VideoFileClip,
        start: float,
        end: float,
        visual_dir: str,
        audio_dir: str,
    ) -> str:
        """
        Separate video into audio and visual.
        :param id:              Sample id.
        :param video:           Video.
        :param start:           Start time.
        :param end:             End time.
        :param visual_dir:      Path to visual output directory.
        :param audio_dir:       Path to audio output directory.
        """
        segment_id = f"{id}-{int(start)}-{int(end)}"
        segment = video.subclip(start, end)
        self.save_visual_clip(
            segment=segment,
            segment_id=segment_id,
            visual_dir=visual_dir,
        )
        self.save_audio_clip(
            segment=segment,
            segment_id=segment_id,
            audio_dir=audio_dir,
        )
        return segment_id

    def save_visual_clip(
        self, segment: mp.VideoFileClip,
        segment_id: str,
        visual_dir: str,
    ) -> None:
        """
        Save video segment.
        :param segment:         Video segment.
        :param segment_id:      Segment id.
        :param visual_dir:      Path to visual output directory.
        """
        visual_path = os.path.join(visual_dir, f"{segment_id}.mp4")
        if not os.path.exists(visual_path):
            segment.without_audio().write_videofile(
                visual_path,
                codec="libx264",
                logger=None,
            )

    def save_audio_clip(
        self, segment: mp.VideoFileClip,
        segment_id: str,
        audio_dir: str,
    ) -> None:
        """
        Save audio segment.
        :param segment:         Video segment.
        :param segment_id:      Segment id.
        :param audio_dir:       Path to audio output directory.
        """
        audio_path = os.path.join(audio_dir, f"{segment_id}.wav")
        if not os.path.exists(audio_path):
            segment.audio.write_audiofile(
                audio_path,
                codec="pcm_s16le",
                logger=None,
            )
