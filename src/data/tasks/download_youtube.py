import os
import argparse
import tqdm
import pytubefix

def args_parser():
    """
    Parse arguments.
    """

    parser = argparse.ArgumentParser(
        description="Download video from tiktok.")
    
    parser.add_argument('--channel_path',           type=str,
                        help='Path list of channels (txt file) - 2 columns (channel_id, num_videos)')
    parser.add_argument('--type',           type=str,
                        help='Download from (Channels | Playlist | Videos)')
    
    parser.add_argument('--overwrite',           action=argparse.BooleanOptionalAction,
                        default=False,  help='Overwrite existing file')
    
    parser.add_argument('--save_path',           type=str,
                        default=None,  help='Path for saving channel')
    
    args = parser.parse_args()

    return args

def download_channel(channel_id: str, save_path: str, num_videos: int = -1):
    """
    Download videos from channel.
    :param channel_id: channel id(Ex: Vietcetera)
    :param save_path: path for saving channel
    :param num_videos: number of videos to download(-1,0: download all videos, >0: download num_videos videos)
    """

    print("-"*100)
    print(f"Downloading videos from channel {channel_id} ...")
    url = f"https://www.youtube.com/c/{channel_id}"

    channel = pytubefix.Channel(url)

    num_download = 0
    if num_videos > 0 and num_videos < len(channel.videos):
        num_download = num_videos
    else:
        num_download = len(channel.videos)

    if not os.path.exists(os.path.join(save_path)):
        os.makedirs(os.path.join(save_path))
    else:
        # check if the channel has been downloaded
        if len(os.listdir(save_path)) == num_download and args.overwrite == False:
            print(f"Channel {channel_id} has been downloaded.")
            return

    pbar = tqdm.tqdm(total=num_download)

    for video in channel.videos:
        try:
            video.streams.get_highest_resolution().download(output_path=save_path, filename=video.video_id)
            pbar.update(1)
            pbar.set_description(f"Downloaded {video}")
            if pbar.n == num_download:
                break
        except:
            print(f"Error when downloading video {video.watch_url} with title {video.title}")
            continue
    

def download_channels(channel_path: str, save_path: str):
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
        download_channel(channel_id, os.path.join(save_path, channel_id), num_videos)
        
if __name__ == "__main__":
    args = args_parser()
    if args.type == "Channels":
        download_channels(args.channel_path, args.save_path)
    elif args.type == "Playlist":
        pass
    elif args.type == "Videos":
        pass
    else:
        print("Type is not supported.")
        exit(1)