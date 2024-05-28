import glob
import math
import os
import time

import numpy as np
import subprocess
import shutil
import pickle
import cv2
import python_speech_features
import torch

from scipy import signal
from scipy.io import wavfile
from scipy.interpolate import interp1d
from typing import Tuple
from huggingface_hub import HfFileSystem

from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

from src.data.processors.processor import Processor
from src.data.processors.uploader import Uploader

from ..utils.Light_ASD.model.ASD import ASD
from ..utils.Light_ASD.model.faceDetector import S3FD

fs = HfFileSystem()


class ActiveSpeakerExtracter(Processor):
    def __init__(self,
                 minTrack: int = 10,
                 numFailedDet: int = 10,
                 facedetScale: float = 0.25,
                 cropScale: float = 0.4,
                 minFaceSize: int = 1,
                 nDataLoaderThread: int = 10,
                 pretrainModel: str = "src/data/utils/Light_ASD/weight/pretrain_AVA_CVPR.model",
                 videoFolderPath: str = None,
                 start: int = 0,
                 duration: int = 0,
                 save: bool = True,  # False
                 clear: bool = False,  # False
                 extract: str = 'origin',
                 ) -> None:

        super().__init__()

        self.minTrack = minTrack
        self.numFailedDet = numFailedDet
        self.facedetScale = facedetScale
        self.cropScale = cropScale
        self.minFaceSize = minFaceSize
        self.nDataLoaderThread = nDataLoaderThread
        self.pretrainModel = pretrainModel
        self.videoFolderPath = videoFolderPath
        self.start = start
        self.duration = duration
        self.save = save
        self.clear = clear
        self.speaking_frame_count_threshold = 70
        self.speaking_score_threshold = 0.8
        self.frame_window_length = 4

        if extract not in ['origin', 'sample']:
            raise ValueError('extract must be origin or sample')
        self.extract = extract

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.facedetScale = facedetScale
        self.DET = S3FD(device=self.device)

        # init for face trạcking
        self.iouThres = 0.5  # Minimum IOU between consecutive face detections

        self.asd = ASD()
        self.asd.loadParameters(self.pretrainModel)

        # init path
        self.pyaviPath = None
        self.pyframesPath = None
        self.pyworkPath = None
        self.pycropPath = None
        self.pyactivePath = None
        self.videoFilePath = None
        self.AudioFilePath = None
        self.network_dir = None
        self.outputPath = None
        self.savePath = None
        self.network_repo_id = "GSU24AI03-SU24AI21/network-result-asd"
        
    def scene_detect(self) -> list:
        # CPU: Scene detection, output is the list of each shot's time duration
        videoManager = VideoManager([self.videoFilePath])
        statsManager = StatsManager()
        sceneManager = SceneManager(statsManager)
        sceneManager.add_detector(ContentDetector())
        baseTimecode = videoManager.get_base_timecode()
        videoManager.set_downscale_factor()
        videoManager.start()
        sceneManager.detect_scenes(frame_source=videoManager)
        sceneList = sceneManager.get_scene_list(baseTimecode)
        if sceneList == []:
            sceneList = [(videoManager.get_base_timecode(),
                          videoManager.get_current_timecode())]
        if self.save:
            with open(os.path.join(self.pyworkPath, 'scene.pckl'), 'wb') as fil:
                pickle.dump(sceneList, fil)

        return sceneList

    def inference_video(self, flist: list, conf_th: float) -> list:
        # GPU: Face detection, output is the list contains the face location and score in this frame
        dets = []
        n_f = len(flist)
        for fidx, fname in enumerate(flist):
            image = cv2.imread(fname)
            imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bboxes = self.DET.detect_faces(
                imageNumpy, conf_th=conf_th, scales=[self.facedetScale])
            dets.append([])
            for bbox in bboxes:
                dets[-1].append({'frame': fidx, 'bbox': (bbox[:-1]).tolist(), 'conf': bbox[-1]})
            print('\rframe: %0.5d/%d' % (fidx, n_f),end='')
        print()
        if self.save:
            with open(os.path.join(self.pyworkPath, 'faces.pckl'), 'wb') as fil:
                pickle.dump(dets, fil)
        return dets

    def bb_intersection_over_union(self, boxA: list, boxB: list) -> float:
        # CPU: IOU Function to calculate overlap between two image
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def track_shot(self, sceneFaces: list) -> list:
        # CPU: Face tracking
        tracks = []
        while True:
            track = []
            for frameFaces in sceneFaces:
                for face in frameFaces:
                    if track == []:
                        track.append(face)
                        frameFaces.remove(face)
                    elif face['frame'] - track[-1]['frame'] <= self.numFailedDet:
                        iou = self.bb_intersection_over_union(
                            face['bbox'], track[-1]['bbox'])
                        if iou > self.iouThres:
                            track.append(face)
                            frameFaces.remove(face)
                            continue
                    else:
                        break
            if track == []:
                break
            elif len(track) > self.minTrack:
                frameNum = np.array([f['frame'] for f in track])
                bboxes = np.array([np.array(f['bbox']) for f in track])
                frameI = np.arange(frameNum[0], frameNum[-1] + 1)
                bboxesI = []
                for ij in range(0, 4):
                    interpfn = interp1d(frameNum, bboxes[:, ij])
                    bboxesI.append(interpfn(frameI))
                bboxesI = np.stack(bboxesI, axis=1)
                if max(np.mean(bboxesI[:, 2] - bboxesI[:, 0]),
                       np.mean(bboxesI[:, 3] - bboxesI[:, 1])) > self.minFaceSize:
                    tracks.append({'frame': frameI, 'bbox': bboxesI})
        return tracks

    def crop_video(self, flist: list, track: dict, cropFile: str) -> dict:
        # CPU: crop the face clips
        vOut = cv2.VideoWriter(
            cropFile + 't.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (224, 224))  # Write video
        dets = {'x': [], 'y': [], 's': []}
        for det in track['bbox']:  # Read the tracks
            dets['s'].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
            dets['y'].append((det[1] + det[3]) / 2)  # crop center x
            dets['x'].append((det[0] + det[2]) / 2)  # crop center y
        dets['s'] = signal.medfilt(
            dets['s'], kernel_size=13)  # Smooth detections
        dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
        dets['y'] = signal.medfilt(dets['y'], kernel_size=13)
        for fidx, frame in enumerate(track['frame']):
            cs = self.cropScale
            bs = dets['s'][fidx]  # Detection box size
            bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount
            image = cv2.imread(flist[frame])
            frame = np.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)),'constant', constant_values=(110, 110))
            my = dets['y'][fidx] + bsi  # BBox center Y
            mx = dets['x'][fidx] + bsi  # BBox center X
            face = frame[int(my - bs):int(my + bs * (1 + 2 * cs)),
                   int(mx - bs * (1 + cs)):int(mx + bs * (1 + cs))]
            vOut.write(cv2.resize(face, (224, 224)))
        audioTmp = cropFile + '.wav'
        audioStart = (track['frame'][0]) / 25
        audioEnd = (track['frame'][-1] + 1) / 25
        vOut.release()
        command = (
                "ffmpeg -y -i %s -async 1 -ac 1 -vn -c:a pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic" %
                (self.audioFilePath, self.nDataLoaderThread, audioStart, audioEnd, audioTmp))
        subprocess.call( command, shell=True, stdout=None)

        command = ("ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a copy %s.avi -loglevel panic" %
                   (cropFile, audioTmp, self.nDataLoaderThread, cropFile))  # Combine audio and video file
        output = subprocess.call(command, shell=True, stdout=None)
        os.remove(cropFile + 't.avi')
        return {'track': track, 'proc_track': dets}

    def evaluate_network(self, files: list) -> list:
        # GPU: active speaker detection by pretrained TalkNet
        self.asd.eval()
        allScores = []
        # durationSet = {1,2,4,6} # To make the result more reliable
        # Use this line can get more reliable result
        durationSet = {1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6}
        for file in files:
            fileName = os.path.splitext(file.split('/')[-1])[0]  # Load audio and video
            _, audio = wavfile.read(os.path.join(
                self.pycropPath, fileName + '.wav'))
            audioFeature = python_speech_features.mfcc(
                audio, 16000, numcep=13, winlen=0.025, winstep=0.010)
            video = cv2.VideoCapture(os.path.join(
                self.pycropPath, fileName + '.avi'))
            videoFeature = []
            while video.isOpened():
                ret, frames = video.read()
                if ret == True:
                    face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
                    face = cv2.resize(face, (224, 224))
                    face = face[int(112 - (112 / 2)):int(112 + (112 / 2)),
                           int(112 - (112 / 2)):int(112 + (112 / 2))]
                    videoFeature.append(face)
                else:
                    break
            video.release()
            videoFeature = np.array(videoFeature)
            length = min(
                (audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0] / 25)
            audioFeature = audioFeature[:int(round(length * 100)), :]
            videoFeature = videoFeature[:int(round(length * 25)), :, :]
            allScore = []  # Evaluation use TalkNet
            for duration in durationSet:
                batchSize = int(math.ceil(length / duration))
                scores = []
                with torch.no_grad():
                    for i in range(batchSize):
                        inputA = torch.FloatTensor(
                            audioFeature[i * duration * 100:(i + 1) * duration * 100, :]).unsqueeze(0).to(device=self.device)
                        inputV = torch.FloatTensor(
                            videoFeature[i * duration * 25: (i + 1) * duration * 25, :, :]).unsqueeze(0).to(device=self.device)
                        embedA = self.asd.model.forward_audio_frontend(inputA)
                        embedV = self.asd.model.forward_visual_frontend(inputV)
                        out = self.asd.model.forward_audio_visual_backend(embedA, embedV)
                        score = self.asd.lossAV.forward(out, labels=None)
                        scores.extend(score)
                allScore.append(scores)
            allScore = np.round(np.mean(np.array(allScore), axis=0), 1).astype(float)
            allScores.append(allScore)
        if self.save:
            with open(os.path.join(self.pyworkPath, 'scores.pckl'), 'wb') as fil:
                pickle.dump(allScores, fil)
        return allScores

    def _get_active_speaker(self, network_dir: str,) -> int:
        work_path: str = os.path.join(network_dir, 'pywork')
        with open(os.path.join(work_path, 'scores.pckl'), mode='rb') as f:
            scores = pickle.load(f)

        speaking_track_paths = []
        for tidx, score in enumerate(scores):
            n_speaking_score = 0
            for idx in range(len(score)):
                sc = score[max(idx - self.frame_window_length, 0): min(idx + self.frame_window_length,len(score) - 1)]
                sc = np.mean(sc)
                if sc > self.speaking_score_threshold:
                    n_speaking_score += 1
            if n_speaking_score > self.speaking_frame_count_threshold:
                track_path = os.path.join('pycrop', '%0.5d.avi' % tidx)
                speaking_track_paths.append(track_path)
        if speaking_track_paths:
            with open(os.path.join(network_dir, 'join_active.txt'), mode='w') as f:
                for track_path in speaking_track_paths:
                    line = 'file {}\n'.format(track_path)
                    f.write(line)
            return 1
        else:
            return 0

    def _merge_videos(self, network_dir: str) -> None:
        join_file = os.path.join(network_dir, 'join_active.txt')
        output_path = os.path.join(network_dir, 'active.avi')
        command = \
            "ffmpeg -y -f concat -safe 0 -i %s -qscale:v 0 -qscale:a 0 -threads %d -r 25 -ar 16000 -c:a copy -c:v copy -f avi %s -loglevel panic" % \
            (join_file, self.nDataLoaderThread,output_path)
        subprocess.call(command, shell=True, stdout=None)

    def _split_into_equally(self, network_dir: str, time_interval: int,) -> Tuple[list,list]:
        active_video = os.path.join(network_dir, 'active.avi')
        network_name = os.path.basename(network_dir)
        channel = network_name.split('@')[0]
        video_id = network_name.split('@')[1]
        chunk_visual_dir = os.path.join(self.outputPath, 'visual', channel)
        chunk_audio_dir = os.path.join(self.outputPath, 'audio', channel)

        os.makedirs(chunk_visual_dir, exist_ok=True)
        os.makedirs(chunk_audio_dir, exist_ok=True)

        command = "ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 %s" % \
                  (active_video,)

        duration = int(float(subprocess.run(command, shell=True, capture_output=True).stdout.decode('utf-8').strip())//3*3)
        if duration < time_interval:
            return [], []
        chunk_visual_ids = []
        chunk_audio_ids = []
        for i, timestamp in enumerate(range(0, duration, time_interval)):
            chunk_visual_id = 'chunk@visual@%s@%s@%05d' % (channel, video_id, i)
            chunk_visual_ids.append(chunk_visual_id)
            chunk_visual_path = os.path.join(chunk_visual_dir, chunk_visual_id + '.mp4')
            command = "ffmpeg -y -i %s -an -c:v copy -r 25 -ss %s -t %d -map 0 -f mp4 %s -loglevel panic" % \
                      (active_video, time.strftime("%H:%M:%S", time.gmtime(timestamp)), time_interval, chunk_visual_path)
            subprocess.run(command, shell=True, stdout=None)

            chunk_audio_id = 'chunk@audio@%s@%s@%05d' % (channel, video_id, i)
            chunk_audio_ids.append(chunk_audio_id)
            chunk_audio_path = os.path.join(chunk_audio_dir, chunk_audio_id + '.wav')
            command = "ffmpeg -y -i %s -vn -ac 1 -c:a pcm_s16le -ar 16000 -ss %s -t %f -f wav %s -loglevel panic" % \
                      (active_video, time.strftime("%H:%M:%S", time.gmtime(timestamp)), time_interval, chunk_audio_path)
            subprocess.run(command, shell=True, stdout=None)

        return chunk_visual_ids, chunk_audio_ids

    def process(
            self,
            sample: dict,
            output_dir: str,
            visual_output_dir: str,
            audio_output_dir: str,
            tmp_dir: str,
            **kwargs
    ) -> dict:
        videoPath = sample['video_path'][0]
        video_file_name = videoPath.split('/')[-1]
        channel = video_file_name.split('@')[0]
        video_id = video_file_name.split('@')[-1][:-4]
        self.outputPath = output_dir
        self.network_dir = os.path.join(tmp_dir, channel, f"{channel}@{video_id}@network_results")
        self.pyaviPath = os.path.join(self.network_dir, 'pyavi')
        self.pyframesPath = os.path.join(self.network_dir, 'pyframes')
        self.pyworkPath = os.path.join(self.network_dir, 'pywork')
        self.pycropPath = os.path.join(self.network_dir, 'pycrop')
        chunk_visual_list = glob.glob(os.path.join(visual_output_dir,f"chunk@visual@{channel}@{video_id}@*.mp4"), recursive=False)
        chunk_audio_list = glob.glob(os.path.join(audio_output_dir,f"chunk@audio@{channel}@{video_id}@*.wav"), recursive=False)
        if chunk_visual_list and chunk_audio_list and len(chunk_visual_list) == len(chunk_audio_list):
            visual_ids = [os.path.basename(pa)[:-4] for pa in chunk_visual_list]
            audio_ids = [os.path.basename(pa)[:-4] for pa in chunk_audio_list]
        else:
            repo_zip_file_network = os.path.join(f"datasets/{self.network_repo_id}@main",channel,f"{channel}@{video_id}@network_results.zip")
            if fs.isfile(repo_zip_file_network):
                local_zip_file_network = self.network_dir + ".zip"
                fs.get(rpath=repo_zip_file_network,lpath=local_zip_file_network)
                shutil.unpack_archive(filename=local_zip_file_network,extract_dir=os.path.join(tmp_dir,channel),format='zip')
                os.remove(local_zip_file_network)
            else:
                os.makedirs(self.pyaviPath, exist_ok=True)
                os.makedirs(self.pyframesPath, exist_ok=True)
                os.makedirs(self.pyworkPath, exist_ok=True)
                os.makedirs(self.pycropPath, exist_ok=True)

            # Extract video
            self.videoFilePath = os.path.join(self.pyaviPath, 'video.avi')
            if not os.path.isfile(self.videoFilePath):
                if sample['uploader'] == 'truyenhinhnhandantv':
                    self.start = 0.
                    self.duration = 25.
                if self.duration == 0:
                    command = ("ffmpeg -y -i %s -c:v libx264 -c:a pcm_s16le -b:v 3000k -b:a 192k -qscale:v 0 -qscale:a 0 "
                                "-r 25 -ar 16000 -threads %d -async 1 %s -loglevel panic " %
                                (videoPath, self.nDataLoaderThread, self.videoFilePath))
                    subprocess.call(command, shell=True, stdout=None)
                else:
                    command = (
                                "ffmpeg -y -i %s -qscale:v 0 -threads %d -ss %.3f -to %.3f -async 1 -r 25 -ar 16000 %s -loglevel panic" %
                                (videoPath, self.nDataLoaderThread, self.start, self.start + self.duration,self.videoFilePath))
                    subprocess.call(command, shell=True, stdout=None)

            # Extract the video frames
            if not os.listdir(self.pyframesPath):
                command = ("ffmpeg -y -i %s -qscale:v 0 -threads %d -f image2 %s -loglevel panic" %
                            (self.videoFilePath, self.nDataLoaderThread, os.path.join(self.pyframesPath, '%06d.jpg')))
                subprocess.call(command, shell=True, stdout=None)

            # Extract audio
            self.audioFilePath = os.path.join(self.pyaviPath, 'audio.wav')
            if not os.path.isfile(self.audioFilePath):
                command = ("ffmpeg -y -i %s -qscale:a 0 -ac 1 -c:a copy -vn -threads %d -ar 16000 %s -loglevel panic" %
                            (self.videoFilePath, self.nDataLoaderThread, self.audioFilePath))
                subprocess.call(command, shell=True, stdout=None)

            out_path = os.path.join(self.pyworkPath, 'scene.pckl')
            if not os.path.isfile(out_path):
                scene = self.scene_detect()
            else:
                with open(out_path,mode='rb') as f:
                    scene = pickle.load(f)

            flist = glob.glob("%s/*.jpg" % self.pyframesPath)
            flist.sort()
            out_path = os.path.join(self.pyworkPath, 'faces.pckl')
            if not os.path.isfile(out_path):
                faces = self.inference_video(flist=flist, conf_th=0.99)
            else:
                with open(out_path, mode='rb') as f:
                    faces = pickle.load(f)

            out_path = os.path.join(self.pyworkPath, 'tracks.pckl')
            if not os.path.isfile(out_path):
                allTracks, vidTracks = [], []
                for shot in scene:
                    # Discard the shot frames less than minTrack frames
                    if shot[1].frame_num - shot[0].frame_num >= self.minTrack:
                        allTracks.extend(self.track_shot(faces[shot[0].frame_num:shot[1].frame_num]))
                # Face clips cropping
                for ii, track in enumerate(allTracks):
                    vidTracks.append(self.crop_video(flist=flist, track=track, cropFile=os.path.join(self.pycropPath, '%05d' % ii)))
                if self.save:
                    with open(os.path.join(self.pyworkPath, 'tracks.pckl'), 'wb') as fil:
                        pickle.dump(vidTracks, fil)

            out_path = os.path.join(self.pyworkPath, 'scores.pckl')
            if not os.path.isfile(out_path):
                files = glob.glob("%s/*.avi" % self.pycropPath)
                files.sort()
                self.evaluate_network(files=files)

            return_code = self._get_active_speaker(self.network_dir)
            visual_ids = []
            audio_ids = []
            if return_code:
                self._merge_videos(self.network_dir)
                visual_ids, audio_ids = self._split_into_equally(self.network_dir, time_interval=3)
            if not visual_ids:
                sample['id'] = [None]
                visual_ids = ['No id']
                audio_ids = ['No id']
        
        if os.path.isdir(self.network_dir):
            Uploader().zip_and_upload_dir(
                dir_path=self.network_dir,
                path_in_repo=os.path.join(channel,f"{channel}@{video_id}@network_results.zip"),
                repo_id=self.network_repo_id,
                overwrite=False,
            )
            shutil.rmtree(self.network_dir)
            shutil.rmtree(self.network_dir)

        out_sample =  {
            "id": sample["id"] * len(visual_ids),
            "channel": sample["channel"] * len(visual_ids),
            "chunk_visual_id": visual_ids,
            "chunk_audio_id": audio_ids,
            "visual_num_frames": [25*3] * len(visual_ids),
            "audio_num_frames": [16000*3] * len(visual_ids),
            "visual_fps": [25] * len(visual_ids),
            "audio_fps": [16000] * len(visual_ids),
        }

        return out_sample