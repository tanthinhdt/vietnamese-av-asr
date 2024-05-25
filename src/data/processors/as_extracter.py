import sys
import time
import os
from typing import Any
sys.path.append(os.getcwd())

import torch
import glob
import subprocess
import warnings
import cv2
import pickle
import numpy
import math
import python_speech_features

from scipy import signal
from shutil import rmtree
from scipy.io import wavfile
from scipy.interpolate import interp1d


from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

from phdkhanh2507.testVLR.utils.TalkNet.talkNet import talkNet
from phdkhanh2507.testVLR.utils.TalkNet.model.faceDetector.s3fd import S3FD
from phdkhanh2507.testVLR.utils.Light_ASD.model.faceDetector.s3fd import S3FD
from phdkhanh2507.testVLR.utils.Light_ASD.model.ASD import ASD_Model



from phdkhanh2507.testVLR.processors.processor import Processor

class ActiveSpeakerExtractor(Processor):
    def __init__(self,
                 minTrack: int = 10,
                 numFailedDet: int = 10,
                 outputPath: str = None,
                 device: str = 'cpu',
                 facedetScale: float = 0.25,
                 cropScale: float = 0.4,
                 minFaceSize: int = 1,
                 nDataLoaderThread: int = 10,
                 pretrainModel: str = "phdkhanh2507/testVLR/utils/TalkNet/pretrain_TalkSet.model",
                 videoFolderPath: str = None,
                 start: int = 0,
                 duration: int = 0,
                 save: bool = True, # False
                 extract: str = 'origin',
                 ) -> None:
        
        super().__init__()

        self.device = device
        self.minTrack = minTrack
        self.numFailedDet = numFailedDet
        self.outputPath = outputPath
        self.facedetScale = facedetScale
        self.cropScale = cropScale
        self.minFaceSize = minFaceSize
        self.nDataLoaderThread = nDataLoaderThread
        self.pretrainModel = pretrainModel
        self.videoFolderPath = videoFolderPath
        self.start = start
        self.duration = duration
        self.save = save

        if extract not in ['origin', 'sample']:
            raise ValueError('extract must be origin or sample')
        self.extract = extract
        
        # init for face detection
        self.facedetScale = facedetScale
        self.DET = S3FD(device=self.device)

        # init for face trạcking
        self.iouThres = 0.5     # Minimum IOU between consecutive face detections

        # init for TalkNet
        self.talkNet = talkNet()
        self.talkNet.loadParameters(self.pretrainModel)

        # init path
        self.pyaviPath = None
        self.pyframesPath = None
        self.pyworkPath = None
        self.pycropPath = None
        self.pyactivePath = None
        self.videoFilePath = None
        self.AudioFilePath = None

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
        for fidx, fname in enumerate(flist):
            image = cv2.imread(fname)
            imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bboxes = self.DET.detect_faces(
                imageNumpy, conf_th=conf_th, scales=[self.facedetScale])
            dets.append([])
            for bbox in bboxes:
                # dets has the frames info, bbox info, conf info
                dets[-1].append({'frame': fidx, 'bbox': (bbox[:-1]
                                                         ).tolist(), 'conf': bbox[-1]})
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
                frameNum = numpy.array([f['frame'] for f in track])
                bboxes = numpy.array([numpy.array(f['bbox']) for f in track])
                frameI = numpy.arange(frameNum[0], frameNum[-1]+1)
                bboxesI = []
                for ij in range(0, 4):
                    interpfn = interp1d(frameNum, bboxes[:, ij])
                    bboxesI.append(interpfn(frameI))
                bboxesI = numpy.stack(bboxesI, axis=1)
                if max(numpy.mean(bboxesI[:, 2]-bboxesI[:, 0]), numpy.mean(bboxesI[:, 3]-bboxesI[:, 1])) > self.minFaceSize:
                    tracks.append({'frame': frameI, 'bbox': bboxesI})
        return tracks

    def crop_video(self, flist: list, track: dict, cropFile: str) -> dict:
        # CPU: crop the face clips
        vOut = cv2.VideoWriter(
            cropFile + 't.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (224, 224))  # Write video
        dets = {'x': [], 'y': [], 's': []}
        for det in track['bbox']:  # Read the tracks
            dets['s'].append(max((det[3]-det[1]), (det[2]-det[0]))/2)
            dets['y'].append((det[1]+det[3])/2)  # crop center x
            dets['x'].append((det[0]+det[2])/2)  # crop center y
        dets['s'] = signal.medfilt(
            dets['s'], kernel_size=13)  # Smooth detections
        dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
        dets['y'] = signal.medfilt(dets['y'], kernel_size=13)
        for fidx, frame in enumerate(track['frame']):
            cs = self.cropScale
            bs = dets['s'][fidx]   # Detection box size
            bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount
            image = cv2.imread(flist[frame])
            frame = numpy.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)),
                              'constant', constant_values=(110, 110))
            my = dets['y'][fidx] + bsi  # BBox center Y
            mx = dets['x'][fidx] + bsi  # BBox center X
            face = frame[int(my-bs):int(my+bs*(1+2*cs)),
                         int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
            vOut.write(cv2.resize(face, (224, 224)))
        audioTmp = cropFile + '.wav'
        audioStart = (track['frame'][0]) / 25
        audioEnd = (track['frame'][-1]+1) / 25
        vOut.release()
        command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic" %
                   (self.audioFilePath, self.nDataLoaderThread, audioStart, audioEnd, audioTmp))
        output = subprocess.call(
            command, shell=True, stdout=None)  # Crop audio file
        _, audio = wavfile.read(audioTmp)
        command = ("ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a copy %s.avi -loglevel panic" %
                   (cropFile, audioTmp, self.nDataLoaderThread, cropFile))  # Combine audio and video file
        output = subprocess.call(command, shell=True, stdout=None)
        os.remove(cropFile + 't.avi')
        return {'track': track, 'proc_track': dets}

    def evaluate_network(self, files : list) -> list:
        # GPU: active speaker detection by pretrained TalkNet
        self.talkNet.eval()
        allScores = []
        # durationSet = {1,2,4,6} # To make the result more reliable
        # Use this line can get more reliable result
        durationSet = {1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6}
        for file in files:
            fileName = os.path.splitext(file.split(
                '/')[-1])[0]  # Load audio and video
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
                    face = face[int(112-(112/2)):int(112+(112/2)),
                                int(112-(112/2)):int(112+(112/2))]
                    videoFeature.append(face)
                else:
                    break
            video.release()
            videoFeature = numpy.array(videoFeature)
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
                            audioFeature[i * duration * 100:(i+1) * duration * 100, :]).unsqueeze(0).cpu()
                        inputV = torch.FloatTensor(
                            videoFeature[i * duration * 25: (i+1) * duration * 25, :, :]).unsqueeze(0).cpu()
                        embedA = self.talkNet.model.forward_audio_frontend(
                            inputA)
                        embedV = self.talkNet.model.forward_visual_frontend(
                            inputV)
                        embedA, embedV = self.talkNet.model.forward_cross_attention(
                            embedA, embedV)
                        out = self.talkNet.model.forward_audio_visual_backend(
                            embedA, embedV)
                        score = self.talkNet.lossAV.forward(out, labels=None)
                        scores.extend(score)
                allScore.append(scores)
            allScore = numpy.round(
                (numpy.mean(numpy.array(allScore), axis=0)), 1).astype(float)
            allScores.append(allScore)
        if self.save:
            with open(os.path.join(self.pyworkPath, 'scores.pckl'), 'wb') as fil:
                pickle.dump(allScores, fil)
        return allScores

    def visualization(self, flist: list, tracks: tuple, scores: list) -> None:
        # CPU: visulize the result for video format
        faces = [[] for i in range(len(flist))]
        for tidx, track in enumerate(tracks):
            score = scores[tidx]
            for fidx, frame in enumerate(track['track']['frame'].tolist()):
                # average smoothing
                s = score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)]
                s = numpy.mean(s)
                faces[frame].append({'track': tidx, 'score': float(s), 's': track['proc_track']['s']
                                    [fidx], 'x': track['proc_track']['x'][fidx], 'y': track['proc_track']['y'][fidx]})
        firstImage = cv2.imread(flist[0])
        fw = firstImage.shape[1]
        fh = firstImage.shape[0]
        vOut = cv2.VideoWriter(os.path.join(
            self.pyaviPath, 'video_only.avi'), cv2.VideoWriter_fourcc(*'XVID'), 25, (fw, fh))
        colorDict = {0: 0, 1: 255}
        for fidx, fname in enumerate(flist):
            image = cv2.imread(fname)
            for face in faces[fidx]:
                clr = colorDict[int((face['score'] >= 0))]
                txt = round(face['score'], 1)
                cv2.rectangle(image, (int(face['x']-face['s']), int(face['y']-face['s'])), (int(
                    face['x']+face['s']), int(face['y']+face['s'])), (0, clr, 255-clr), 10)
                cv2.putText(image, '%s' % (txt), (int(face['x']-face['s']), int(
                    face['y']-face['s'])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, clr, 255-clr), 5)
            vOut.write(image)
        vOut.release()
        command = ("ffmpeg -y -i %s -i %s -threads %d -c:v copy -c:a copy %s -loglevel panic" %
                   (os.path.join(self.pyaviPath, 'video_only.avi'), os.path.join(self.pyaviPath, 'audio.wav'),
                    self.nDataLoaderThread, os.path.join(self.pyaviPath, 'video_out.avi')))
        output = subprocess.call(command, shell=True, stdout=None)

    def extract_active_speaker(
            self,
            flist: list,
            tracks: tuple,
            scores: list,
            videoName : str,

    ) -> None:
        # CPU: visulize the result for video format
        for tidx, track in enumerate(tracks):
            active_file = os.path.join(
                self.pyactivePath, videoName + '%05d' % tidx)
            score = scores[tidx]
            active_speaker_frame = []
            longest_active_speaker_frame = []

            for fidx, frame in enumerate(track['track']['frame'].tolist()):
                # average smoothing
                s = score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)]
                s = numpy.mean(s)
                if s > 0:
                    active_speaker_frame.append((fidx, frame))
                else:
                    if len(active_speaker_frame) > len(longest_active_speaker_frame):
                        longest_active_speaker_frame = active_speaker_frame
                    active_speaker_frame = []

            if len(active_speaker_frame) >= len(longest_active_speaker_frame):
                longest_active_speaker_frame = active_speaker_frame
            if len(longest_active_speaker_frame) != 0 and len(longest_active_speaker_frame) > self.minTrack:
                vOut = cv2.VideoWriter(
                    active_file + 't.avi', cv2.VideoWriter_fourcc(*'XVID'),  25, (224, 224))
                for fidx, frame in longest_active_speaker_frame:
                    cs = self.cropScale
                    bs = track['proc_track']['s'][fidx]   # Detection box size
                    bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount
                    image = cv2.imread(flist[frame])
                    frame = numpy.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)),
                                      'constant', constant_values=(110, 110))
                    my = track['proc_track']['y'][fidx] + bsi  # BBox center Y
                    mx = track['proc_track']['x'][fidx] + bsi  # BBox center X
                    face = frame[int(my-bs):int(my+bs*(1+2*cs)),
                                 int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
                    vOut.write(cv2.resize(face, (224, 224)))
                vOut.release()

                audioTmp = active_file + '.wav'
                audioStart = (longest_active_speaker_frame[0][1]) / 25
                audioEnd = (longest_active_speaker_frame[-1][1]+1) / 25
                vOut.release()
                command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic" %
                           (self.audioFilePath, self.nDataLoaderThread, audioStart, audioEnd, audioTmp))
                output = subprocess.call(
                    command, shell=True, stdout=None)  # Crop audio file
                _, audio = wavfile.read(audioTmp)
                command = ("ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a copy %s.avi -loglevel panic" %
                           (active_file, audioTmp, self.nDataLoaderThread, active_file))  # Combine audio and video file
                output = subprocess.call(command, shell=True, stdout=None)
                os.remove(active_file + 't.avi')
                os.remove(audioTmp)
            else:
                continue

    def process(
        self,
        sample: dict,
    ) -> None:
        try:
            videoPath = sample['video_path']
            videoName = os.path.splitext(os.path.basename(videoPath))[0]
            savePath = os.path.join(self.outputPath, videoName,'network_results')

            if os.path.exists(savePath):
                rmtree(savePath)

            self.pyaviPath = os.path.join(savePath, 'pyavi')
            self.pyframesPath = os.path.join(savePath, 'pyframes')
            self.pyworkPath = os.path.join(savePath, 'pywork')
            self.pycropPath = os.path.join(savePath, 'pycrop')
            self.pyactivePath = os.path.join(savePath, 'pyactive')
            # The path for the input video, input audio, output video
            os.makedirs(self.pyaviPath, exist_ok=True)
            # Save all the video frames
            os.makedirs(self.pyframesPath, exist_ok=True)
            # Save the results in this process by the pckl method
            os.makedirs(self.pyworkPath, exist_ok=True)
            # Save the detected face clips (audio+video) in this process
            os.makedirs(self.pycropPath, exist_ok=True)
            # save video with active speaker
            os.makedirs(self.pyactivePath, exist_ok=True)

            # Extract video
            self.videoFilePath = os.path.join(self.pyaviPath, 'video.avi')

            # If duration did not set, extract the whole video, otherwise extract the video from 'self.start' to 'self.start + self.duration'
            if self.duration == 0:
                command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -async 1 -r 25 %s -loglevel panic" %
                            (videoPath, self.nDataLoaderThread, self.videoFilePath))
            else:
                command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -ss %.3f -to %.3f -async 1 -r 25 %s -loglevel panic" %
                            (videoPath, self.nDataLoaderThread, self.start, self.start + self.duration, self.videoFilePath))
            subprocess.call(command, shell=True, stdout=None)

            # Extract audio
            self.audioFilePath = os.path.join(self.pyaviPath, 'audio.wav')
            command = ("ffmpeg -y -i %s -qscale:a 0 -ac 1 -vn -threads %d -ar 16000 %s -loglevel panic" %
                        (self.videoFilePath, self.nDataLoaderThread, self.audioFilePath))
            subprocess.call(command, shell=True, stdout=None)

            # Extract the video frames
            command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -f image2 %s -loglevel panic" %
                        (self.videoFilePath, self.nDataLoaderThread, os.path.join(self.pyframesPath, '%06d.jpg')))
            subprocess.call(command, shell=True, stdout=None)

            
            # Scene detection for the video frames
            scene = self.scene_detect()

            # Get the frame list
            flist = glob.glob(os.path.join(self.pyframesPath, '*.jpg'))
            flist.sort()

            # Face detection for the video frames
            faces = self.inference_video(flist=flist)

            # Face tracking
            allTracks, vidTracks = [], []
            for shot in scene:
                # Discard the shot frames less than minTrack frames
                if shot[1].frame_num - shot[0].frame_num >= self.minTrack:
                    # 'frames' to present this tracks' timestep, 'bbox' presents the location of the faces
                    allTracks.extend(self.track_shot(
                        faces[shot[0].frame_num:shot[1].frame_num]))

            # Face clips cropping
            for ii, track in enumerate(allTracks):
                vidTracks.append(self.crop_video(
                    flist=flist, track=track, cropFile=os.path.join(self.pycropPath, '%05d' % ii)))

            if self.save:
                with open(os.path.join(self.pyworkPath, 'tracks.pckl'), 'wb') as fil:
                    pickle.dump(vidTracks, fil)
            
            files = glob.glob("%s/*.avi" % self.pycropPath)
            files.sort()
            # Active Speaker Detection by TalkNet
            scores = self.evaluate_network(files=files)


            if self.extract == 'origin':
                self.visualization(
                    flist=flist, tracks=vidTracks, scores=scores)
            elif self.extract == 'sample':
                self.extract_active_speaker(
                    flist=flist, tracks=vidTracks, scores=scores, videoName=videoName)

            # delete the tmp files
            dirs_to_remove = []
            if self.save:
                dirs_to_remove += [self.pyframesPath, self.pycropPath]
            else:
                dirs_to_remove += [self.pyworkPath,
                                    self.pyframesPath, self.pycropPath]
            if self.extract == "origin":
                dirs_to_remove += [self.pyactivePath]
            if self.extract == "sample":
                dirs_to_remove += [self.pyaviPath]

            # Remove the directories.
            for dir_to_remove in dirs_to_remove:
                rmtree(dir_to_remove)

            # rename save path
            os.rename(savePath, savePath + "_finished")
        except Exception as e:
            print(e)
            print("Error processing video: %s" % videoPath)
            os.rename(savePath, savePath + "_failed")
    
if __name__ == "__main__":
    videoFolderPath = 'phdkhanh2507/testVLR/video/susupham1406'
    outputPath = 'phdkhanh2507/testVLR/extract/susupham1406'
    activeSpeakerExtract = ActiveSpeakerExtractor(videoFolderPath=videoFolderPath,
                                                  outputPath=outputPath,
                                                  extract="sample")
    # run in multi-process
    file_list = glob.glob("%s/*.mp4" % videoFolderPath)
    # multi_pool = torch.multiprocessing.Pool(processes=os.cpu_count())
    # multi_pool.map(activeSpeakerExtract.process, file_list)
    # multi_pool.close()
    # multi_pool.join()
    activeSpeakerExtract.process(file_list)