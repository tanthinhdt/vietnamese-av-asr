import os
import sys
sys.path.append(os.getcwd())

import glob
import torch
import subprocess
import argparse
import warnings
import shutil
import torch.multiprocessing as mp

import moviepy.editor as mv
from datasets import Dataset, get_dataset_config_names
from huggingface_hub import HfApi
import tqdm
from logging import getLogger

from phdkhanh2507.testVLR.processors.as_extracter import ActiveSpeakerExtractor

logger = getLogger(__name__)
warnings.filterwarnings("ignore")


def args_parser():
    """
    Extract face and active speaker from raw video arguments.
    """

    parser = argparse.ArgumentParser(
        description="Extract face and active speaker from raw video arguments.")
    
    parser.add_argument('--video_folder_path',           type=str,
                        default="phdkhanh2507/testVLR/inputs/*",  help='Path for inputs')
    
    parser.add_argument('--video_extension',           type=str,
                        default="mp4",  help='extension of video files [mp4, avi]')
    
    parser.add_argument('--output_folder',           type=str,
                        default="phdkhanh2507/testVLR/video/susupham1406",  help='Path for temps, outputs')
    
    parser.add_argument('--pretrain_model',         type=str,
                        default="/home/khanhphd/Documents/data/phdkhanh2507/testVLR/utils/TalkNet/pretrain_TalkSet.model",   help='Path for the pretrained TalkNet model')

    parser.add_argument('--n_datalader_thread',     type=int,
                        default=10,   help='Number of workers')
    
    parser.add_argument('--num_proc',           type=int,
                        default=None,  help='Number of processes')
    
    parser.add_argument('--facedet_scale',          type=float, 
                        default=0.25,  help='Scale factor for face detection, the frames will be scale to 0.25 orig')
    
    parser.add_argument('--min_track',              type=int,
                        default=10,   help='Number of min frames for each shot')
    
    parser.add_argument('--num_failed_det',          type=int,   
                        default=10,   help='Number of missed detections allowed before tracking is stopped')
    
    parser.add_argument('--min_face_size',           type=int,
                        default=1,    help='Minimum face size in pixels')
    
    parser.add_argument('--crop_scale',             type=float,
                        default=0.40, help='Scale bounding box')

    parser.add_argument('--start',                 type=int,
                        default=0,   help='The start time of the video')
    
    parser.add_argument('--extract',                 type=str,
                        default='sample',   help='the way to extract active speaker [sample, origin]')
    
    parser.add_argument('--duration',              type=int, default=0,
                        help='The duration of the video, when set as 0, will extract the whole video')

    args = parser.parse_args()

    if os.path.isfile(args.pretrain_model) == False: # Download the pretrained model
        Link = "1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea"
        cmd = "gdown --id %s -O %s"%(Link, args.pretrain_model)
        subprocess.call(cmd, shell=True, stdout=None)
    
    if args.num_proc is None:
        args.num_proc = os.cpu_count()

    args.activespeaker = ActiveSpeakerExtractor(
        minTrack=args.min_track,
        numFailedDet=args.num_failed_det,
        minFaceSize=args.min_face_size,
        cropScale=args.crop_scale,
        start=args.start,
        duration=args.duration,
        pretrainModel=args.pretrain_model,
        facedetScale=args.facedet_scale,
        nDataLoaderThread=args.n_datalader_thread,
        videoFolderPath=args.video_folder_path,
        outputPath=args.output_folder,
        extract=args.extract,
    )

    return args


def get_dataset(data_dir: str, data_out : str, video_extension: str):

    source = glob.glob(data_dir + "/*" + video_extension)
    source = [os.path.basename(x).split(".")[0] for x in source]
    out = glob.glob(data_out + "/*")
    out = [os.path.basename(x) for x in out]
    unfinished = source.copy()
    for file in source:
        if file + "_finished" in out:
            unfinished.remove(file)
    unfinished = [os.path.join(data_dir, x + "." + video_extension) for x in unfinished]
    return list(unfinished)

def generate_sample(channel_dir):
    """
    Generate sample for each channel for uploading to hf.
    :param channel_dir:     channel directory.
    :return:                sample.
    """
    ids = []
    channels = [os.path.basename(channel_dir)] # [:-7]
    fps = []
    sampling_rate = []

    new_channel_dir = channel_dir # [:-7]
    os.makedirs(new_channel_dir, exist_ok=True)

    for chunk_dir in glob.glob(os.path.join(channel_dir, "*")):
        for file_path in glob.glob(os.path.join(chunk_dir, "pyactive", "*.avi")):
            # Extract id
            ids.append(os.path.basename(file_path).split(".")[0])

            video = mv.VideoFileClip(file_path)
            # Extract fps
            fps.append(int(video.fps))
            # Extract sampling rate
            sampling_rate.append(video.audio.fps)
            # Move file
            new_file_path = os.path.join(new_channel_dir, os.path.basename(file_path))
            shutil.move(file_path, new_file_path)
        shutil.rmtree(chunk_dir)
            
    # Remove empty directory
    # shutil.rmtree(channel_dir)

    return {
        "id": ids,
        "channel": channels * len(ids),
        "fps": fps,
        "sampling_rate": sampling_rate,
    }

def upload_to_hf(channel_name, active_speaker_dir, token):
    api = HfApi()
    new_channel_name = os.path.basename(channel_name) #[:-7]
    if new_channel_name in get_dataset_config_names("phdkhanh2507/vietnamese-speaker-video"):
        return

    # Generate metadata
    dataset = Dataset.from_dict(
        generate_sample(os.path.join(active_speaker_dir, "video", channel_name))
    )

    # dataset = Dataset.from_dict(
    #     generate_sample(os.path.join(active_speaker_dir, "extract", channel_name))
    # )

    # Save metadata
    metadata_path = os.path.join(active_speaker_dir, "metadata", new_channel_name + ".parquet")
    dataset.to_parquet(metadata_path)

    # Upload to huggingface
    api.upload_file(
        path_or_fileobj=metadata_path,
        path_in_repo=f"metadata/{new_channel_name}.parquet",
        repo_id="phdkhanh2507/vietnamese-speaker-video",
        repo_type="dataset",
        commit_message="chore: update dataset metadata",
        commit_description=f"Add {channel_name}",
        token=token,
    )

    new_channel_dir = os.path.join(active_speaker_dir, "video", new_channel_name)
    # Zip folder
    shutil.make_archive(
        new_channel_dir, "zip", os.path.dirname(new_channel_dir), os.path.basename(new_channel_dir)
    )
    
    # Upload to huggingface
    api.upload_file(
        path_or_fileobj=new_channel_dir + ".zip",
        path_in_repo=f"video/{new_channel_name}.zip",
        repo_id="phdkhanh2507/vietnamese-speaker-video",
        repo_type="dataset",
        commit_message="chore: update dataset video",
        commit_description=f"Add {new_channel_name}",
        token=token,
    )
    
    # Remove zip file
    os.remove(os.path.join(new_channel_dir + ".zip"))

def main(args):
    """
    Main function.
    :param args:    arguments.
    """
    # Prepare dataset.
    logger.info("Preparing dataset...")
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    
    # un_file_list = get_dataset(args.video_folder_path, args.output_folder, args.video_extension)
    # unf_files_num = len(un_file_list)

    # if unf_files_num == 0:
    #     print('Nothing to do.')
    #     exit(0)
    # print('Found {} unprocessed files.'.format(unf_files_num))
    # print('Processing with {} processes.'.format(args.num_proc))

    # # Count possible number of processes.
    # mp.set_start_method('spawn')
    # with tqdm.tqdm(total=unf_files_num) as pbar:
    #     def update(*a):
    #         pbar.update()

    #     with torch.multiprocessing.Pool(processes=args.num_proc) as pool:
    #         for _ in pool.imap_unordered(args.activespeaker.process, un_file_list):
    #             update()
    # # close pool
    # pool.close()
    # pool.join()

    # remove folder not contain "_finished"
    for file in glob.glob(args.output_folder + "/*"):
        if "_finished" not in file:
            shutil.rmtree(file)
    # get activate speaker folder
    active_speaker_dir = os.path.split(os.path.split(args.output_folder)[0])[0]
    # upload to hf
    upload_to_hf(os.path.basename(args.output_folder), active_speaker_dir, token = "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")


if __name__ == "__main__":
    args = args_parser()
    main(args)    