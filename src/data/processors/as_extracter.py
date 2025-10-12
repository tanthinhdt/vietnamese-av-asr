import copy
import glob
import math
import os
import numpy as np
import subprocess
import shutil
import pickle
import cv2
import python_speech_features
import torch
import mediapipe as mdp

from scipy              import signal
from scipy.io           import wavfile
from scipy.interpolate  import interp1d
from typing             import List, Tuple
from huggingface_hub    import HfFileSystem
from collections        import OrderedDict

from scenedetect.video_manager  import VideoManager
from scenedetect.scene_manager  import SceneManager
from scenedetect.stats_manager  import StatsManager
from scenedetect.detectors      import ContentDetector

from src.data.processors.processor  import Processor
from src.data.processors.uploader   import Uploader
from src.data.utils.Light_ASD.model import ASD
from src.models.utils               import get_logger, get_spent_time

fs = HfFileSystem()


class ActiveSpeakerExtracter(Processor):
    """This class used to detect active speaker in video."""

    def __init__(self,
                 minTrack: int = 10,
                 numFailedDet: int = 10,
                 facedetScale: float = 0.25,
                 cropScale: float = 0.4,
                 minFaceSize: int = 1,
                 nDataLoaderThread: int = 10,
                 pretrainModel: str = "src/weights/pretrain_AVA_CVPR.model",
                 ) -> None:

        super().__init__()

        self.minTrack               = minTrack
        self.numFailedDet           = numFailedDet
        self.facedetScale           = facedetScale
        self.cropScale              = cropScale
        self.minFaceSize            = minFaceSize
        self.nDataLoaderThread      = nDataLoaderThread
        self.iouThres               = 0.5
        self.pretrainModel          = pretrainModel
        self.device                 = "cuda" if torch.cuda.is_available() else "cpu"
        self.face_conf_threshold    = 0.5

        # threshold
        self.speaking_frame_count_threshold     = 30
        self.speaking_score_threshold           = 0.6
        self.frame_window_length                = 4
        self.time_interval                      = 3

        # hyper data
        self.DURATION   = 3
        self.V_FPS      = 25
        self.A_FPS      = 16000

        self.facedetScale           = facedetScale
        self.mp_face_detection      = mdp.solutions.face_detection
        self.short_range_detector   = self.mp_face_detection.FaceDetection(
            min_detection_confidence=self.face_conf_threshold, model_selection=0)
        self.full_range_detector    = self.mp_face_detection.FaceDetection(
            min_detection_confidence=self.face_conf_threshold, model_selection=1)

        self.asd = ASD()
        self.asd.loadParameters(self.pretrainModel)

        # init path
        self.pycropPath         = None
        self.pyoriginPath       = None
        self.pyworkPath         = None
        self.videoFilePath      = None
        self.audioFilePath      = None
        self.network_dir        = None
        self.outputPath         = None
        self.network_repo_id    = "GSU24AI03-SU24AI21/network-result-asd"

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

    def _get_frames(self, video_file: str) -> dict:
        cap = cv2.VideoCapture(video_file)

        i = 0
        frames = OrderedDict()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames[i] = frame
            i += 1
        cap.release()
        return frames

    def _get_duration(self, video_file: str):
        command = "ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 %s" % (
            video_file,)
        try:
            duration = int(float(subprocess.run(command, shell=True, capture_output=True).stdout.strip()))
        except Exception:
            duration = -1
        return duration

    def _get_scene(self) -> List[dict]:
        out_path = os.path.join(self.pyworkPath, 'scene.pckl')
        if not os.path.isfile(out_path):
            scene = self.scene_detect()
        else:
            with open(out_path,mode='rb') as f:
                scene = pickle.load(f)
        return scene

    def scene_detect(self) -> list:
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
        with open(os.path.join(self.pyworkPath, 'scene.pckl'), 'wb') as fil:
            pickle.dump(sceneList, fil)

        return sceneList

    def _get_faces(self) -> Tuple[dict, list]:
        out_path = os.path.join(self.pyworkPath, 'faces.pckl')
        if not os.path.isfile(out_path):
            frames, faces = self._detect_faces(video_path=self.videoFilePath)
        else:
            with open(out_path, mode='rb') as f:
                faces = pickle.load(f)
            frames = self._get_frames(video_file=self.videoFilePath)

        return frames, faces

    @get_spent_time(message="Detect face in second:")
    def _detect_faces(self, video_path: str) -> Tuple[dict, list]:
        frames = OrderedDict()
        dets = []
        cap = cv2.VideoCapture(video_path)
        n_f = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fidx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            ih, iw = frame.shape[:2]
            imageNumpy = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames[fidx] = frame
            results = self.full_range_detector.process(imageNumpy)
            if not results.detections:
                results = self.short_range_detector.process(imageNumpy)
            dets.append([])
            if results.detections:
                for detection in results.detections:
                    if detection.score[0] > self.face_conf_threshold:
                        bboxC = detection.location_data.relative_bounding_box
                        bbox = (bboxC.xmin * iw, bboxC.ymin * ih, (bboxC.width + bboxC.xmin) * iw, (bboxC.height + bboxC.ymin) * ih)
                        dets[-1].append({'frame': fidx, 'bbox': bbox, 'conf': detection.score[0]})
            print('\rInfering frame: %0.5d/%d' % (fidx, n_f), end='')
            fidx += 1
        cap.release()
        print()
        with open(os.path.join(self.pyworkPath, 'faces.pckl'), 'wb') as fil:
            pickle.dump(dets, fil)
        return frames, dets

    def _filter_out_duration(self, video_file: str):
        _skip = False
        duration = -1
        if os.path.isfile(video_file):
            duration = self._get_duration(video_file)
            if duration < self.DURATION:
                _skip = True

        if _skip:
            # remove audio file
            os.remove(path=video_file)

            audio_file = video_file.replace('avi', 'wav')
            if os.path.isfile(audio_file):
                os.remove(path=audio_file)

            # remove origin video corresponding
            origin_file = video_file.replace(self.pycropPath, self.pyoriginPath)
            if os.path.isfile(origin_file):
                os.remove(path=origin_file)

            # remove origin audio corresponding
            origin_file = origin_file.replace('avi', 'wav')
            if os.path.isfile(origin_file):
                os.remove(path=origin_file)

        return _skip, duration

    def _crop_face(self, scene: list, faces: list, frames: dict, keep_origin: bool = False) -> None:
        sort_keys = sorted(frames.keys())
        _frames = OrderedDict({k: frames[k] for k in sort_keys})
        out_path = os.path.join(self.pyworkPath, 'tracks.pckl')
        if not os.path.isfile(out_path):
            allTracks, vidTracks = [], []
            for shot in scene:
                if shot[1].frame_num - shot[0].frame_num >= self.minTrack:
                    allTracks.extend(self.track_shot(faces[shot[0].frame_num:shot[1].frame_num]))
            ii = 0
            for track in allTracks:
                crop_name = os.path.join(self.pycropPath, '%05d' % ii)
                origin_name = os.path.join(self.pyoriginPath, '%05d' % ii)

                vidTracks.append(self.crop_video(frames=_frames, track=track, cropFile=crop_name))
                if keep_origin:
                    self.truncate_video(frames=_frames, track=track, trunc_file=origin_name)

                if self._filter_out_duration(video_file=crop_name + '.avi')[0]:
                    vidTracks.pop(-1)
                else:
                    ii += 1

            with open(os.path.join(self.pyworkPath, 'tracks.pckl'), 'wb') as fil:
                pickle.dump(vidTracks, fil)

    def track_shot(self, sceneFaces: list) -> list:
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

    def crop_video(self, frames: dict, track: dict, cropFile: str) -> None:
        vOut = cv2.VideoWriter(cropFile + 't.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (224, 224))
        dets = {'x': [], 'y': [], 's': []}
        for det in track['bbox']:  # Read the tracks
            dets['s'].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
            dets['y'].append((det[1] + det[3]) / 2)  # crop center x
            dets['x'].append((det[0] + det[2]) / 2)  # crop center y
        dets['s'] = signal.medfilt(dets['s'], kernel_size=13)  # Smooth detections
        dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
        dets['y'] = signal.medfilt(dets['y'], kernel_size=13)
        for fidx, frame in enumerate(track['frame']):
            cs      = self.cropScale
            bs      = dets['s'][fidx]  # Detection box size
            bsi     = int(bs * (1 + 2 * cs))  # Pad videos by this amount
            image   = frames[frame]
            frame   = np.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)),'constant', constant_values=(110, 110))
            my      = dets['y'][fidx] + bsi  # BBox center Y
            mx      = dets['x'][fidx] + bsi  # BBox center X
            face    = frame[int(my - bs):int(my + bs * (1 + 2 * cs)),
                            int(mx - bs * (1 + cs)):int(mx + bs * (1 + cs))]
            vOut.write(cv2.resize(face, (224, 224)))
        audioTmp    = cropFile + '.wav'
        audioStart  = (track['frame'][0]) / 25
        audioEnd    = (track['frame'][-1] + 1) / 25
        vOut.release()
        command = (
                "ffmpeg -y -i %s -async 1 -ac 1 -vn -c:a pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic" %
                (self.videoFilePath, self.nDataLoaderThread, audioStart, audioEnd, audioTmp))
        subprocess.call(command, shell=True, stdout=None)

        command = ("ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a copy %s.avi -loglevel panic" %
                   (cropFile, audioTmp, self.nDataLoaderThread, cropFile))
        subprocess.call(command, shell=True, stdout=None)

        os.remove(cropFile + 't.avi')
        return {'track': track, 'proc_track': dets}

    def truncate_video(self, track: dict, trunc_file: str) -> None:
        start_time = (track['frame'][0]) / 25
        end_time = (track['frame'][-1] + 1) / 25

        command = ("ffmpeg -y -i %s -threads %d -c:v libx264 -c:a pcm_s16le -ss %.3f -to %.3f -f avi %s.avi -loglevel panic" %
                   (self.videoFilePath, self.nDataLoaderThread, start_time, end_time, trunc_file))
        subprocess.call(command, shell=True, stdout=None)

    def _compute_scores(self) -> None:
        out_path = os.path.join(self.pyworkPath, 'scores.pckl')
        if not os.path.isfile(out_path):
            files = glob.glob("%s/*.avi" % self.pycropPath)
            files.sort()
            self.evaluate_network(files=files)

    @get_spent_time(message="Evaluate score in second:")
    def evaluate_network(self, files: list) -> None:
        self.asd.eval()
        allScores = []
        # durationSet = {1,2,4,6} # To make the result more reliable
        durationSet = {1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6}
        for file in files:
            fileName = os.path.splitext(file.split('/')[-1])[0]
            _, audio = wavfile.read(os.path.join(self.pycropPath, fileName + '.wav'))
            audioFeature = python_speech_features.mfcc(
                audio, 16000, numcep=13, winlen=0.025, winstep=0.010)
            video = cv2.VideoCapture(os.path.join(self.pycropPath, fileName + '.avi'))
            videoFeature = []
            while video.isOpened():
                ret, frames = video.read()
                if ret:
                    face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
                    face = cv2.resize(face, (224, 224))
                    face = face[int(112 - (112 / 2)):int(112 + (112 / 2)),int(112 - (112 / 2)):int(112 + (112 / 2))]
                    videoFeature.append(face)
                else:
                    break
            video.release()
            videoFeature = np.array(videoFeature)
            length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0] / 25)
            audioFeature = audioFeature[:int(round(length * 100)), :]
            videoFeature = videoFeature[:int(round(length * 25)), :, :]
            allScore = []
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
        with open(os.path.join(self.pyworkPath, 'scores.pckl'), 'wb') as fil:
            pickle.dump(allScores, fil)

    def _detect_active_speaker(self) -> list:
        with open(os.path.join(self.pyworkPath, 'scores.pckl'), mode='rb') as f:
            scores = pickle.load(f)
        _crop_paths = []
        for tidx, score in enumerate(scores):
            n_speaking_frame = 0
            scs = []
            for idx in range(len(score)):
                sc = score[max(idx - self.frame_window_length, 0): min(idx + self.frame_window_length, len(score) - 1)]
                sc = np.mean(sc)
                if sc > self.speaking_score_threshold:
                    n_speaking_frame += 1
                    scs.append(sc)
            if n_speaking_frame > self.speaking_frame_count_threshold:
                _crop_path = os.path.join(self.pycropPath, '%0.5d.avi' % tidx)
                _crop_paths.append(_crop_path)

        return _crop_paths

    def _split_into_equally(self, crop_paths: list, combine_av: bool = False, keep_origin: bool = False) -> Tuple[list, list]:
        network_name        = os.path.basename(self.network_dir)
        channel             = network_name.split('@')[1]
        video_id            = network_name.split('@')[2]
        chunk_visual_dir    = os.path.join(self.outputPath, 'visual', channel)
        chunk_audio_dir     = os.path.join(self.outputPath, 'audio', channel)
        chunk_video_dir     = os.path.join(self.outputPath, 'video', channel)
        origin_video_dir    = os.path.join(self.outputPath, 'origin', channel)
        os.makedirs(chunk_visual_dir, exist_ok=True)
        os.makedirs(chunk_audio_dir, exist_ok=True)
        os.makedirs(chunk_video_dir, exist_ok=True)
        os.makedirs(origin_video_dir, exist_ok=True)

        chunk_visual_ids    = []
        chunk_audio_ids     = []
        i = 0
        for _crop_path in crop_paths:
            duration = math.ceil(self._get_duration(video_file=_crop_path) // self.time_interval * self.time_interval)
            if duration < 0:
                continue
            for timestamp in range(0, duration, self.time_interval):
                start_time = float(timestamp)
                end_time = float(timestamp + self.time_interval)

                chunk_visual_id     = 'chunk@visual@%s@%s@%05d' % (channel, video_id, i)
                chunk_visual_path   = os.path.join(chunk_visual_dir, chunk_visual_id + '.mp4')
                command = "ffmpeg -y -i %s -an -c:v libx264 -b:v 1200k -r 25 -ss %.3f -to %.3f -map 0 -f avi %s -loglevel panic" % \
                        (_crop_path, start_time, end_time, chunk_visual_path)
                subprocess.run(command, shell=True, stdout=None)

                chunk_audio_id      = 'chunk@audio@%s@%s@%05d' % (channel, video_id, i)
                chunk_audio_path    = os.path.join(chunk_audio_dir, chunk_audio_id + '.wav')
                command = "ffmpeg -y -i %s -vn -ac 1 -c:a pcm_s16le -ar 16000 -b:a 192k -ss %.3f -to %.3f -map 0 -f wav %s -loglevel panic" % \
                        (_crop_path, start_time, end_time, chunk_audio_path)
                subprocess.run(command, shell=True, stdout=None)

                if combine_av:
                    # combine chunk visual an chunk audio
                    chunk_video_id = 'chunk@video@%s@%s@%05d' % (channel, video_id, i)
                    chunk_video_path = os.path.join(chunk_video_dir, chunk_video_id + '.mp4')
                    command = "ffmpeg -y -i %s -i %s -c:v copy -c:a copy -map 0:v:0 -map 1:a:0 %s -loglevel panic" % \
                              (chunk_visual_path, chunk_audio_path, chunk_video_path)
                    subprocess.run(command, shell=True, stdout=None)

                if keep_origin:
                    # truncate origin video corresponding with cropped face
                    origin_video_id = 'chunk@origin@%s@%s@%05d' % (channel, video_id, i)
                    origin_video_path = os.path.join(origin_video_dir, origin_video_id + '.mp4')
                    command = "ffmpeg -y -i %s -c:v libx264 -c:a aac -ss %.3f -to %.3f -map 0:v:0 -map 0:a:0 -f avi %s -loglevel panic" % \
                              (_crop_path.replace('pycrop', 'pyorigin'), start_time, end_time, origin_video_path)
                    subprocess.run(command, shell=True, stdout=None)

                if self._check_output(
                    visual_path=chunk_visual_path,
                    audio_path=chunk_audio_path
                ):
                    chunk_visual_ids.append(chunk_visual_id)
                    chunk_audio_ids.append(chunk_audio_id)
                    i += 1
        return chunk_visual_ids, chunk_audio_ids,

    def _make_network_result(self, tmp_dir: str, channel: str, video_id: str, demo: bool = False) -> bool:
        _up = True
        repo_zip_file_network = os.path.join(
            f"datasets/{self.network_repo_id}@main",
            "network_results", channel,
            f"network_results@{channel}@{video_id}.zip"
        )
        if fs.isfile(repo_zip_file_network) and not demo:
            print("Downloading network results...")
            local_zip_file_network = self.network_dir + ".zip"
            fs.get(rpath=repo_zip_file_network,lpath=local_zip_file_network)
            shutil.unpack_archive(filename=local_zip_file_network,extract_dir=os.path.join(tmp_dir,channel),format='zip')
            os.remove(local_zip_file_network)
            _up = False
            _old_nw = glob.glob(os.path.join(tmp_dir,channel,f'*{video_id}*'),recursive=False)[0]
            if _old_nw != self.network_dir:
                os.rename(src=_old_nw,dst=self.network_dir)
                _up = True
            [os.remove(_redun_file) for _redun_file in glob.glob(self.network_dir+'/*active.*',recursive=False)]
        else:
            os.makedirs(self.pycropPath, exist_ok=True)
            os.makedirs(self.pyoriginPath, exist_ok=True)
            os.makedirs(self.pyworkPath, exist_ok=True)
            _up = False if demo else _up
        return _up

    def _extract_video(self, origin_video_path: str) -> None:
        self.videoFilePath = origin_video_path
        self.audioFilePath = self.videoFilePath

        
    def process(
            self,
            sample: dict,
            output_dir: str,
            visual_output_dir: str,
            audio_output_dir: str,
            tmp_dir: str,
            time_interval: int = 3,
            log_path: str = None,
            face_conf_threshold: float = 0,
            **kwargs
    ) -> dict:
        """
        Detect speaker in video.

        sample:
            Dict contains metadata.
        output_dir:
            Directory container processed clip.
        visual_output_dir:
            Directory container processed visual.
        visual_output_dir:
            Directory container processed audio.
        tmp_dir:
            Directory contains asd network result.
        log_path:
            Path of log file.
        return:
            Metadat of processed sample.
        """
        combine_av = kwargs.get('combine_av', False)
        keep_origin = kwargs.get('keep_origin', False)
        clear_nw = kwargs.get('clear_nw', True)

        logger = get_logger(
            name=__name__,
            log_path=log_path,
            is_stream=True,
        )
        logger_ = get_logger(
            log_path=log_path,
            is_stream=True,
            format="%(message)s"
        )

        videoPath   = sample['video_path'][0]
        channel     = sample['channel'][0]
        video_id    = sample['video_id'][0]
        
        logger_.info('-'*36 + f"AS-detector processing video id '{video_id}'" + '-'*36)
        self.face_conf_threshold = face_conf_threshold
        self.time_interval  = time_interval
        self.DURATION       = time_interval
        self.outputPath     = output_dir
        self.network_dir    = os.path.join(tmp_dir, channel, f"network_results@{channel}@{video_id}")
        self.pyworkPath     = os.path.join(self.network_dir, 'pywork')
        self.pycropPath     = os.path.join(self.network_dir, 'pycrop')
        self.pyoriginPath   = os.path.join(self.network_dir, 'pyorigin')

        chunk_visual_list   = glob.glob(os.path.join(visual_output_dir, f"chunk@visual@{channel}@{video_id}@*.mp4"), recursive=False)
        chunk_audio_list    = glob.glob(os.path.join(audio_output_dir, f"chunk@audio@{channel}@{video_id}@*.wav"), recursive=False)
        if chunk_visual_list and chunk_audio_list and len(chunk_visual_list) == len(chunk_audio_list):
            visual_ids      = [os.path.basename(pa)[:-4] for pa in chunk_visual_list]
            audio_ids       = [os.path.basename(pa)[:-4] for pa in chunk_audio_list]
        else:
            logger.info("Check nw")
            up = self._make_network_result(tmp_dir=tmp_dir, channel=channel, video_id=video_id, demo=sample['demo'][0])

            logger.info("Extract video")
            self._extract_video(origin_video_path=videoPath)

            logger.info('Get scene')
            scene = self._get_scene()

            logger.info('Get faces')
            frames, faces = self._get_faces()

            logger.info('Crop faces')
            self._crop_face(scene=scene, faces=faces, frames=frames, keep_origin=keep_origin)

            logger.info('Evaluate scores')
            self._compute_scores()

            logger.info('Detect active speaker')
            crop_paths  = self._detect_active_speaker()
            visual_ids  = []
            audio_ids   = []
            if crop_paths:
                logger.info(f'Split into {self.time_interval}s segment')
                visual_ids, audio_ids = self._split_into_equally(
                    crop_paths=crop_paths,
                    combine_av=combine_av,
                    keep_origin=keep_origin,
                )
            if not visual_ids:
                sample['id']    = [None]
                visual_ids      = ['None']
                audio_ids       = ['None']

            if os.path.isdir(self.network_dir):
                if up:
                    logger.info('Upload nw')
                    Uploader().zip_and_upload_dir(
                        dir_path=self.network_dir,
                        path_in_repo=os.path.join("network_results", channel, f"network_results@{channel}@{video_id}.zip"),
                        repo_id=self.network_repo_id,
                        overwrite=False,
                    )
                if not sample['demo'] or clear_nw:
                    prefix, _ = os.path.split(self.network_dir.rstrip('/'))
                    shutil.rmtree(prefix)

        output_sample: dict = copy.copy(sample)
        for k in sample.keys():
            if k != 'id':
                output_sample.pop(k)

        output_sample["id"]                    = sample["id"] * len(visual_ids)
        output_sample["channel"]               = sample["channel"] * len(audio_ids)
        output_sample["chunk_visual_id"]       = visual_ids
        output_sample["chunk_audio_id"]        = audio_ids
        output_sample["visual_num_frames"]     = [self.V_FPS * self.DURATION] * len(visual_ids)
        output_sample["audio_num_frames"]      = [self.A_FPS * self.DURATION] * len(audio_ids)
        output_sample["visual_fps"]            = [self.V_FPS] * len(visual_ids)
        output_sample["audio_fps"]             = [self.A_FPS] * len(audio_ids)

        logger_.info('*'*50 + 'AS-detector done.' + '*'*50)

        return output_sample
    
    def _check_output(self, visual_path: str, audio_path: str,) -> bool:
        
        _duration_cmd       = "ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 %s"
        _visual_frame_cmd   = "ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of csv=p=0 %s"
        _visual_fps_cmd     = "ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of default=noprint_wrappers=1:nokey=1 %s"
        _audio_fps_cmd      = "ffprobe -v error -select_streams a -show_entries stream=sample_rate -of default=noprint_wrappers=1:nokey=1 %s"
        
        _v_dur = float(subprocess.run(_duration_cmd % (visual_path,), shell=True, capture_output=True).stdout)
        _a_dur = float(subprocess.run(_duration_cmd % (audio_path,), shell=True, capture_output=True).stdout)

        _v_fra = float(subprocess.run(_visual_frame_cmd % (visual_path,), shell=True, capture_output=True).stdout)

        _v_fps = float(subprocess.run(_visual_fps_cmd % (visual_path,), shell=True, capture_output=True).stdout[:2])
        _a_fps = float(subprocess.run(_audio_fps_cmd % (audio_path,), shell=True, capture_output=True).stdout)

        check_true = (_v_dur == self.DURATION and
                      _a_dur == self.DURATION and
                      _v_fra == self.V_FPS * self.DURATION and
                      _a_fps == self.A_FPS and
                      _v_fps == self.V_FPS)

        if not check_true:
            os.remove(visual_path)
            os.remove(audio_path)

        return check_true