import asyncio
import io
import glob
import os
import urllib.request
from os import path
import tqdm
from typing import List

import aiohttp
import argparse
from tiktokapipy.async_api import AsyncTikTokAPI
from tiktokapipy.models.video import Video


def args_parser():
    """
    Parse arguments.
    """

    parser = argparse.ArgumentParser(
        description="Download video from tiktok.")
    
    parser.add_argument('--channel_path',           type=str,
                        default="phdkhanh2507/testVLR/channel.txt",  help='Path list of channels (txt file) - 2 columns (channel_id, num_videos)')
    
    parser.add_argument('--video_url',           type=str,
                        default="phdkhanh2507/testVLR/video_url",  help='Path contain url to download video (txt file)')
    
    parser.add_argument('--overwrite',           action='store_true',
                        default=True,  help='Overwrite existing file')
    
    parser.add_argument('--save_path',           type=str,
                        default="phdkhanh2507/testVLR/video",  help='Path for saving channel')
    
    args = parser.parse_args()

    return args

async def save_slideshow(directory, video: Video):
    """
    This function downloads the images and music from a slideshow and uses ffmpeg to join them together
    :param directory: the directory to save the images and music to
    :param video: the video object to download
    :return: the joined video
    """

    # this filter makes sure the images are padded to all the same size
    vf = r"\"scale=iw*min(1080/iw\,1920/ih):ih*min(1080/iw\,1920/ih)," \
         r"pad=1080:1920:(1080-iw)/2:(1920-ih)/2," \
         r"format=yuv420p\""

    for i, image_data in enumerate(video.image_post.images):
        url = image_data.image_url.url_list[-1]
        # this step could probably be done with asyncio, but I didn't want to figure out how
        urllib.request.urlretrieve(url, path.join(directory, f"temp_{video.id}_{i:02}.jpg"))

    urllib.request.urlretrieve(video.music.play_url, path.join(directory, f"temp_{video.id}.mp3"))

    # use ffmpeg to join the images and audio
    command = [
        "ffmpeg",
        "-r 2/5",
        f"-i {directory}/temp_{video.id}_%02d.jpg",
        f"-i {directory}/temp_{video.id}.mp3",
        "-r 30",
        f"-vf {vf}",
        "-acodec copy",
        f"-t {len(video.image_post.images) * 2.5}",
        f"{directory}/temp_{video.id}.mp4",
        "-y"
    ]
    ffmpeg_proc = await asyncio.create_subprocess_shell(
        " ".join(command),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await ffmpeg_proc.communicate()
    generated_files = glob.glob(path.join(directory, f"temp_{video.id}*"))

    if not path.exists(path.join(directory, f"temp_{video.id}.mp4")):
        # optional ffmpeg logging step
        # logging.error(stderr.decode("utf-8"))
        for file in generated_files:
            os.remove(file)
        raise Exception("Something went wrong with piecing the slideshow together")

    with open(path.join(directory, f"temp_{video.id}.mp4"), "rb") as f:
        ret = io.BytesIO(f.read())

    for file in generated_files:
        os.remove(file)

    return ret


async def save_video(video: Video, api: AsyncTikTokAPI):
    """
    This function downloads a video from TikTok
    :param video: the video object to download
    :param api: the AsyncTikTokAPI instance to use
    :return: the joined video
    """

    # Carrying over this cookie tricks TikTok into thinking this ClientSession was the Playwright instance
    # used by the AsyncTikTokAPI instance
    async with aiohttp.ClientSession(cookies={cookie["name"]: cookie["value"] for cookie in await api.context.cookies() if cookie["name"] == "tt_chain_token"}) as session:
        # Creating this header tricks TikTok into thinking it made the request itself
        async with session.get(video.video.download_addr, headers={"referer": "https://www.tiktok.com/"}) as resp:
            return io.BytesIO(await resp.read())


async def download_video(link, output_path):
    """
    This function downloads a video from TikTok
    :param link: the link to the video
    :param output_path: the directory to save the video to
    """

    async with AsyncTikTokAPI() as api:
        video: Video = await api.video(link)
        if video.image_post:
            downloaded = await save_slideshow(directory=output_path, video=video)
        else:
            downloaded = await save_video(video, api)
        # save the video to a file
        with open(path.join(output_path, f"{video.id}.mp4"), "wb") as f:
            f.write(downloaded.read())
        

# async def down_video_tiktok_from_user(user_id, output_path, video_limit=None):
#     """
#     Download videos from user.
#     :param user_id: user id(Ex: vietcetera)
#     :param output_path: path for saving user
#     :param video_limit: number of videos to download(-1,0: download all videos, >0: download num_videos videos)
#     """

#     async with AsyncTikTokAPI(navigation_timeout=0) as api:
#         if video_limit is not None:
#             user = await api.user(user_id, video_limit=video_limit)
#             pbar = tqdm.tqdm(total=video_limit)
#         else:
#             user = await api.user(user_id)
#             pbar = tqdm.tqdm(total=user.stats.video_count)
#         async for video in user.videos:
#             link = f"https://www.tiktok.com/@{user_id}/video/{video.id}"
#             await download_video(link=link, output_path=output_path)
#             pbar.update(1)
#             pbar.set_description(f"Downloaded {video.id}")
#         pbar.close()
           
async def download_channel(channel_videos: List[str], save_path: str, channel_id: str, num_videos: int = -1):

    num_download = 0
    if num_videos > 0 and num_videos < len(channel_videos):
        num_download = num_videos
    else:
        num_download = len(channel_videos)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        # check if the channel has been downloaded
        if len(os.listdir(save_path)) == num_download and args.overwrite == False:
            print(f"Channel {channel_id} has been downloaded.")
            return
        
    pbar = tqdm.tqdm(total=num_download)

    for video in channel_videos:
        try:
            await download_video(link=video, output_path=save_path)
            pbar.update(1)
            pbar.set_description(f"Downloaded {video}")
            if pbar.n == num_download:
                break
        except Exception:
            print(f"Error when downloading video {video}")
    pbar.close()

    # remove jpg files
    for f in os.listdir(save_path):
        if f.endswith(".jpg"):
            os.remove(os.path.join(save_path, f))

async def download_channels(channel_path: str, save_path: str):
    """
    Download videos from list of channel.
    :param channel_path: path to file containing list of channels
    :param save_path: path for saving channels
    """
    # read channel list
    with open(channel_path, 'r') as f:
        lines = f.readlines()
    channels = [line.strip().split(',') for line in lines]
    channels = [(channel[0], int(channel[1])) for channel in channels]

    for channel_id, num_videos in channels:
        try:
            with open(os.path.join(args.video_url, f"{channel_id}.txt"), 'r') as f:
                lines = f.readlines()
            videos = [line.strip() for line in lines]
            
            await download_channel(videos, os.path.join(save_path, channel_id), channel_id, num_videos)

        except FileNotFoundError:
            print(f"File {channel_id}.txt not found.")
            continue

if __name__ == "__main__":
    args = args_parser()

    if args.channel_path is not None:
        asyncio.run(download_channels(args.channel_path, args.save_path))



    



