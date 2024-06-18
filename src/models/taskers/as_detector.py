import os

from typing import List

from src.models.taskers.tasker import Tasker
from src.models.utils import get_logger
from src.data.processors.as_extracter import ActiveSpeakerExtracter


class ASDetector(Tasker):

    def __init__(self, n_process: int = 1):
        super().__init__()
        self.detector = ActiveSpeakerExtracter(n_process=n_process)
        self.output_dir = 'data/processed'
        self.visual_output_dir = self.output_dir + '/visual/'
        self.audio_output_dir = self.output_dir + '/audio/'
        self.tmp_dir = 'data/interim'

        self._logger = get_logger(name=__name__, is_stream=True)
    # def __init__(self,
    #              minTrack: int = 10,
    #              numFailedDet: int = 10,
    #              facedetScale: float = 0.25,
    #              cropScale: float = 0.4,
    #              minFaceSize: int = 1,
    #              nDataLoaderThread: int = 10,
    #              pretrainModel: str = "src/weights/pretrain_AVA_CVPR.model",
    #              ) -> None:
    #
    #     super().__init__()
    #
    #     self.minTrack = minTrack
    #     self.numFailedDet = numFailedDet
    #     self.facedetScale = facedetScale
    #     self.cropScale = cropScale
    #     self.minFaceSize = minFaceSize
    #     self.nDataLoaderThread = nDataLoaderThread
    #     self.pretrainModel = pretrainModel
    #     self.device = "cuda" if torch.cuda.is_available() else "cpu"
    #
    #     # threshold
    #     self.speaking_frame_count_threshold = 30
    #     self.speaking_score_threshold = 0.6
    #     self.frame_window_length = 4
    #     self.time_interval = 3
    #     self.start = 0.0
    #     self.duration = 10000.0
    #
    #     # hyper data
    #     self.V_FPS = 25
    #     self.A_FPS = 16000
    #     self.DURATION = 3
    #     self.V_FRAMES = self.V_FPS * self.DURATION
    #     self.A_FRAMES = self.A_FPS * self.DURATION
    #
    #     self.facedetScale = facedetScale
    #     self.DET = S3FD(device=self.device)
    #
    #     # init for face trạcking
    #     self.iouThres = 0.5  # Minimum IOU between consecutive face detections
    #
    #     self.asd = ASD()
    #     self.asd.loadParameters(self.pretrainModel)
    #
    #     # init path
    #     self.pyaviPath = None
    #     self.pyframesPath = None
    #     self.pyworkPath = None
    #     self.pycropPath = None
    #     self.videoFilePath = None
    #     self.audioFilePath = None
    #     self.network_dir = None
    #     self.outputPath = None
    #     self.network_repo_id = "GSU24AI03-SU24AI21/network-result-asd"
    #
    # def scene_detect(self) -> list:
    #     # CPU: Scene detection, output is the list of each shot's time duration
    #     videoManager = VideoManager([self.videoFilePath])
    #     statsManager = StatsManager()
    #     sceneManager = SceneManager(statsManager)
    #     sceneManager.add_detector(ContentDetector())
    #     baseTimecode = videoManager.get_base_timecode()
    #     videoManager.set_downscale_factor()
    #     videoManager.start()
    #     sceneManager.detect_scenes(frame_source=videoManager)
    #     sceneList = sceneManager.get_scene_list(baseTimecode)
    #     if not sceneList:
    #         sceneList = [(videoManager.get_base_timecode(),
    #                       videoManager.get_current_timecode())]
    #     with open(os.path.join(self.pyworkPath, 'scene.pckl'), 'wb') as fil:
    #         pickle.dump(sceneList, fil)
    #
    #     return sceneList
    #
    # @get_spent_time
    # def _inference_video(self, flist: list, conf_th: float) -> list:
    #     n_f = len(flist)
    #     dets = []
    #     for fidx, fname in enumerate(flist):
    #         image = cv2.imread(fname)
    #         imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #         bboxes = self.DET.detect_faces(
    #             imageNumpy, conf_th=conf_th, scales=[self.facedetScale])
    #         dets.append([])
    #         for bbox in bboxes:
    #             dets[-1].append({'frame': fidx, 'bbox': (bbox[:-1]).tolist(), 'conf': bbox[-1]})
    #         print('\rInfering frame: %0.5d/%d' % (fidx, n_f), end='')
    #     print()
    #     with open(os.path.join(self.pyworkPath, 'faces.pckl'), 'wb') as fil:
    #         pickle.dump(dets, fil)
    #     return dets
    #
    # @get_spent_time
    # def _inference_video_v2(self, video_path: str,  conf_th: float):
    #     frames = []
    #     dets = []
    #     cap = cv2.VideoCapture(video_path)
    #     n_f = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    #     fidx = 0
    #     while cap.isOpened():
    #         ret, frame = cap.read()
    #         if not ret:
    #             break
    #         frames.append(frame)
    #     for frame in frames:
    #         imageNumpy = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         bboxes = self.DET.detect_faces(
    #             imageNumpy, conf_th=conf_th, scales=[self.facedetScale])
    #         frame_det = []
    #         for bbox in bboxes:
    #             frame_det.append({'frame': fidx, 'bbox': (bbox[:-1]).tolist(), 'conf': bbox[-1]})
    #         dets.append(frame_det)
    #         print('\rDetecting frame: %0.5d/%d' % (fidx, n_f), end='')
    #         fidx += 1
    #     return dets
    #
    # @get_spent_time
    # def _inference_video_v1(self, flist: list, conf_th: float) -> list:
    #     images = [cv2.imread(f) for f in flist]
    #     n_f = len(images)
    #     n_processes = 5
    #     _chunk_size = math.ceil(n_f / n_processes)
    #     _kwargs = []
    #     for i in range(0, n_f, _chunk_size):
    #         _chunks_i = images[i:i + _chunk_size]
    #         _chunks_iidx = range(i, i + _chunk_size)
    #         _kwargs.append(
    #             {
    #                 'images': _chunks_i,
    #                 'iidxs': _chunks_iidx,
    #                 'n_f': n_f,
    #                 'conf_th': conf_th,
    #             }
    #         )
    #
    #     with mp.Pool(processes=n_processes) as pool:
    #         dets: list = pool.map(self._inference_frames, _kwargs)
    #
    #     with open(os.path.join(self.pyworkPath, 'faces.pckl'), 'wb') as fil:
    #         pickle.dump(dets, fil)
    #
    #     return dets
    #
    # def _inference_frames(self, kwargs: dict) -> List:
    #     dets = []
    #     for iidx, image in zip(kwargs['iidxs'], kwargs['images']):
    #         imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #         bboxes = self.DET.detect_faces(
    #             imageNumpy, conf_th=kwargs['conf_th'], scales=[self.facedetScale])
    #         frame_det = []
    #         for bbox in bboxes:
    #             frame_det.append({'frame': iidx, 'bbox': (bbox[:-1]).tolist(), 'conf': bbox[-1]})
    #         dets.append(frame_det)
    #         print('\rDetecting frame: %0.5d/%d' % (iidx, kwargs['n_f']), end='')
    #     return dets
    #
    # def bb_intersection_over_union(self, boxA: list, boxB: list) -> float:
    #     # CPU: IOU Function to calculate overlap between two image
    #     xA = max(boxA[0], boxB[0])
    #     yA = max(boxA[1], boxB[1])
    #     xB = min(boxA[2], boxB[2])
    #     yB = min(boxA[3], boxB[3])
    #     interArea = max(0, xB - xA) * max(0, yB - yA)
    #     boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    #     boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    #     iou = interArea / float(boxAArea + boxBArea - interArea)
    #     return iou
    #
    # def track_shot(self, sceneFaces: list) -> list:
    #     # CPU: Face tracking
    #     tracks = []
    #     while True:
    #         track = []
    #         for frameFaces in sceneFaces:
    #             for face in frameFaces:
    #                 if track == []:
    #                     track.append(face)
    #                     frameFaces.remove(face)
    #                 elif face['frame'] - track[-1]['frame'] <= self.numFailedDet:
    #                     iou = self.bb_intersection_over_union(
    #                         face['bbox'], track[-1]['bbox'])
    #                     if iou > self.iouThres:
    #                         track.append(face)
    #                         frameFaces.remove(face)
    #                         continue
    #                 else:
    #                     break
    #         if track == []:
    #             break
    #         elif len(track) > self.minTrack:
    #             frameNum = np.array([f['frame'] for f in track])
    #             bboxes = np.array([np.array(f['bbox']) for f in track])
    #             frameI = np.arange(frameNum[0], frameNum[-1] + 1)
    #             bboxesI = []
    #             for ij in range(0, 4):
    #                 interpfn = interp1d(frameNum, bboxes[:, ij])
    #                 bboxesI.append(interpfn(frameI))
    #             bboxesI = np.stack(bboxesI, axis=1)
    #             if max(np.mean(bboxesI[:, 2] - bboxesI[:, 0]),
    #                    np.mean(bboxesI[:, 3] - bboxesI[:, 1])) > self.minFaceSize:
    #                 tracks.append({'frame': frameI, 'bbox': bboxesI})
    #     return tracks
    #
    # def crop_video(self, flist: list, track: dict, cropFile: str) -> dict:
    #     # CPU: crop the face clips
    #     vOut = cv2.VideoWriter(
    #         cropFile + 't.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (224, 224))  # Write video
    #     dets = {'x': [], 'y': [], 's': []}
    #     for det in track['bbox']:  # Read the tracks
    #         dets['s'].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
    #         dets['y'].append((det[1] + det[3]) / 2)  # crop center x
    #         dets['x'].append((det[0] + det[2]) / 2)  # crop center y
    #     dets['s'] = signal.medfilt(dets['s'], kernel_size=13)
    #     dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
    #     dets['y'] = signal.medfilt(dets['y'], kernel_size=13)
    #     for fidx, frame in enumerate(track['frame']):
    #         cs = self.cropScale
    #         bs = dets['s'][fidx]  # Detection box size
    #         bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount
    #         image = cv2.imread(flist[frame])
    #         frame = np.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)), 'constant', constant_values=(110, 110))
    #         my = dets['y'][fidx] + bsi  # BBox center Y
    #         mx = dets['x'][fidx] + bsi  # BBox center X
    #         face = frame[int(my - bs):int(my + bs * (1 + 2 * cs)),
    #                int(mx - bs * (1 + cs)):int(mx + bs * (1 + cs))]
    #         vOut.write(cv2.resize(face, (224, 224)))
    #     audioTmp = cropFile + '.wav'
    #     audioStart = (track['frame'][0]) / 25
    #     audioEnd = (track['frame'][-1] + 1) / 25
    #     vOut.release()
    #     command = (
    #             "ffmpeg -y -i %s -async 1 -ac 1 -vn -c:a pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic" %
    #             (self.audioFilePath, self.nDataLoaderThread, audioStart, audioEnd, audioTmp))
    #     subprocess.call(command, shell=True, stdout=None)
    #
    #     command = ("ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a copy %s.avi -loglevel panic" %
    #                (cropFile, audioTmp, self.nDataLoaderThread, cropFile))  # Combine audio and video file
    #     output = subprocess.call(command, shell=True, stdout=None)
    #     os.remove(cropFile + 't.avi')
    #     return {'track': track, 'proc_track': dets}
    #
    # def evaluate_network(self, files: list) -> list:
    #     # GPU: active speaker detection by pretrained TalkNet
    #     self.asd.eval()
    #     allScores = []
    #     # durationSet = {1,2,4,6} # To make the result more reliable
    #     # Use this line can get more reliable result
    #     durationSet = {1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6}
    #     for file in files:
    #         fileName = os.path.splitext(file.split('/')[-1])[0]  # Load audio and video
    #         _, audio = wavfile.read(os.path.join(
    #             self.pycropPath, fileName + '.wav'))
    #         audioFeature = python_speech_features.mfcc(
    #             audio, 16000, numcep=13, winlen=0.025, winstep=0.010)
    #         video = cv2.VideoCapture(os.path.join(self.pycropPath, fileName + '.avi'))
    #         videoFeature = []
    #         while video.isOpened():
    #             ret, frames = video.read()
    #             if ret:
    #                 face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    #                 face = cv2.resize(face, (224, 224))
    #                 face = face[int(112 - (112 / 2)):int(112 + (112 / 2)),int(112 - (112 / 2)):int(112 + (112 / 2))]
    #                 videoFeature.append(face)
    #             else:
    #                 break
    #         video.release()
    #         videoFeature = np.array(videoFeature)
    #         length = min(
    #             (audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0] / 25)
    #         audioFeature = audioFeature[:int(round(length * 100)), :]
    #         videoFeature = videoFeature[:int(round(length * 25)), :, :]
    #         allScore = []  # Evaluation use TalkNet
    #         for duration in durationSet:
    #             batchSize = int(math.ceil(length / duration))
    #             scores = []
    #             with torch.no_grad():
    #                 for i in range(batchSize):
    #                     inputA = torch.FloatTensor(
    #                         audioFeature[i * duration * 100:(i + 1) * duration * 100, :]).unsqueeze(0).to(
    #                         device=self.device)
    #                     inputV = torch.FloatTensor(
    #                         videoFeature[i * duration * 25: (i + 1) * duration * 25, :, :]).unsqueeze(0).to(
    #                         device=self.device)
    #                     embedA = self.asd.model.forward_audio_frontend(inputA)
    #                     embedV = self.asd.model.forward_visual_frontend(inputV)
    #                     out = self.asd.model.forward_audio_visual_backend(embedA, embedV)
    #                     score = self.asd.lossAV.forward(out, labels=None)
    #                     scores.extend(score)
    #             allScore.append(scores)
    #         allScore = np.round(np.mean(np.array(allScore), axis=0), 1).astype(float)
    #         allScores.append(allScore)
    #     with open(os.path.join(self.pyworkPath, 'scores.pckl'), 'wb') as fil:
    #         pickle.dump(allScores, fil)
    #
    # def _detect_active_speaker(self) -> List[str]:
    #     with open(os.path.join(self.pyworkPath, 'scores.pckl'), mode='rb') as f:
    #         scores = pickle.load(f)
    #     _crop_paths = []
    #     for tidx, score in enumerate(scores):
    #         n_speaking_frame = 0
    #         scs = []
    #         for idx in range(len(score)):
    #             sc = score[max(idx - self.frame_window_length, 0): min(idx + self.frame_window_length, len(score) - 1)]
    #             sc = np.mean(sc)
    #             if sc > self.speaking_score_threshold:
    #                 n_speaking_frame += 1
    #                 scs.append(sc)
    #         if n_speaking_frame > self.speaking_frame_count_threshold:
    #             _crop_path = os.path.join(self.pycropPath, '%0.5d.avi' % tidx)
    #             _crop_paths.append(_crop_path)
    #     return _crop_paths
    #
    # def _split_into_equally(self, crop_paths: list, ) -> ...:
    #     chunk_visual_dir = os.path.join(self.outputPath, 'visual')
    #     chunk_audio_dir = os.path.join(self.outputPath, 'audio')
    #     chunk_video_dir = os.path.join(self.outputPath, 'video')
    #     os.makedirs(chunk_visual_dir, exist_ok=True)
    #     os.makedirs(chunk_audio_dir, exist_ok=True)
    #     os.makedirs(chunk_video_dir, exist_ok=True)
    #     i = 0
    #     _chunk_visual_paths = []
    #     _chunk_audio_paths = []
    #     _chunk_video_paths = []
    #     for _crop_path in crop_paths:
    #         if not os.path.isfile(_crop_path):
    #             continue
    #         command = "ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 %s" % (
    #         _crop_path,)
    #         try:
    #             duration = int(float(subprocess.run(command, shell=True, capture_output=True).stdout.strip()) // 3 * 3)
    #             if duration < self.time_interval:
    #                 continue
    #             for timestamp in range(0, duration, self.time_interval):
    #
    #                 _chunk_visual_path = os.path.join(chunk_visual_dir, "visual_%.5d" % i + '.mp4')
    #                 command = "ffmpeg -y -i %s -an -c:v libx264 -ss %d -t %d -map 0 -f mp4 %s -loglevel panic" % \
    #                           (_crop_path, timestamp, self.time_interval, _chunk_visual_path)
    #                 subprocess.run(command, shell=True, stdout=None)
    #
    #                 _chunk_audio_path = os.path.join(chunk_audio_dir, "audio_%.5d" % i + '.wav')
    #                 command = "ffmpeg -y -i %s -vn -ac 1 -c:a pcm_s16le -ss %s -t %s -f wav %s -loglevel panic" % \
    #                           (_crop_path, timestamp, self.time_interval, _chunk_audio_path)
    #                 subprocess.run(command, shell=True, stdout=None)
    #
    #                 _chunk_video_path = os.path.join(chunk_video_dir, "video_%.5d" % i + '.av.mp4')
    #                 command = "ffmpeg -y -i %s -i %s -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 -f mp4 %s -loglevel panic" % \
    #                           (_chunk_visual_path, _chunk_audio_path, _chunk_video_path)
    #                 subprocess.run(command, shell=True, stdout=None)
    #
    #                 if self._check_output(visual_path=_chunk_visual_path, audio_path=_chunk_audio_path):
    #                     _chunk_visual_paths.append(_chunk_visual_path)
    #                     _chunk_audio_paths.append(_chunk_audio_path)
    #                     _chunk_video_paths.append(_chunk_video_path)
    #
    #                 i += 1
    #         except ValueError as e:
    #             continue
    #
    #     return _chunk_visual_paths, _chunk_audio_paths, _chunk_video_paths
    #
    # def _get_scene(self) -> List[dict]:
    #     out_path = os.path.join(self.pyworkPath, 'scene.pckl')
    #     if not os.path.isfile(out_path):
    #         scene = self.scene_detect()
    #     else:
    #         with open(out_path, mode='rb') as f:
    #             scene = pickle.load(f)
    #     return scene
    #
    # def _get_faces(self) -> List[dict]:
    #     flist = glob.glob("%s/*.jpg" % self.pyframesPath)
    #     flist.sort()
    #     out_path = os.path.join(self.pyworkPath, 'faces.pckl')
    #     if not os.path.isfile(out_path):
    #         faces = self._inference_video(flist=flist, conf_th=0.99)
    #         #faces = self._inference_video_v1(flist=flist, conf_th=0.99)
    #         #faces = self._inference_video_v2(video_path=self.videoFilePath, conf_th=0.99)
    #     else:
    #         with open(out_path, mode='rb') as f:
    #             faces = pickle.load(f)
    #     return faces
    #
    # def _crop_face(self, scene: list, faces: list) -> List[dict]:
    #     flist = glob.glob("%s/*.jpg" % self.pyframesPath)
    #     flist.sort()
    #     out_path = os.path.join(self.pyworkPath, 'tracks.pckl')
    #     if not os.path.isfile(out_path):
    #         allTracks, vidTracks = [], []
    #         for shot in scene:
    #             if shot[1].frame_num - shot[0].frame_num >= self.minTrack:
    #                 allTracks.extend(self.track_shot(faces[shot[0].frame_num:shot[1].frame_num]))
    #         for ii, track in enumerate(allTracks):
    #             vidTracks.append(
    #                 self.crop_video(flist=flist, track=track, cropFile=os.path.join(self.pycropPath, '%05d' % ii)))
    #
    #         with open(os.path.join(self.pyworkPath, 'tracks.pckl'), 'wb') as fil:
    #             pickle.dump(vidTracks, fil)
    #
    # def _compute_scores(self) -> None:
    #     out_path = os.path.join(self.pyworkPath, 'scores.pckl')
    #     if not os.path.isfile(out_path):
    #         files = glob.glob("%s/*.avi" % self.pycropPath)
    #         files.sort()
    #         self.evaluate_network(files=files)
    #
    # def _make_network_result(self):
    #     os.makedirs(self.pyaviPath, exist_ok=True)
    #     os.makedirs(self.pyframesPath, exist_ok=True)
    #     os.makedirs(self.pyworkPath, exist_ok=True)
    #     os.makedirs(self.pycropPath, exist_ok=True)
    #
    # def _extract_video(self, origin_video_path: str) -> None:
    #     self.videoFilePath = os.path.join(self.pyaviPath, 'video.avi')
    #     if not os.path.isfile(self.videoFilePath):
    #         command = ("ffmpeg -y -i %s -c:v libx264 -c:a aac -b:v 3000k -b:a 192k -qscale:v 0 -qscale:a 0 "
    #                    "-r 25 -ar 16000 -ss %.3f -t %.3f -threads %d -async 1 -f mp4 %s -loglevel panic " %
    #                    (origin_video_path, self.start, self.duration, self.nDataLoaderThread, self.videoFilePath))
    #         subprocess.call(command, shell=True, stdout=None)
    #
    # def _extract_frames(self) -> None:
    #     if not os.listdir(self.pyframesPath):
    #         command = ("ffmpeg -y -i %s -qscale:v 0 -threads %d -f image2 %s -loglevel panic" %
    #                    (self.videoFilePath, self.nDataLoaderThread, os.path.join(self.pyframesPath, '%06d.jpg')))
    #         subprocess.call(command, shell=True, stdout=None)
    #
    # def _extract_audio(self) -> None:
    #     self.audioFilePath = os.path.join(self.pyaviPath, 'audio.wav')
    #     if not os.path.isfile(self.audioFilePath):
    #         command = ("ffmpeg -y -i %s -qscale:a 0 -ac 1 -c:a pcm_s16le -vn -threads %d -ar 16000 -f wav %s -loglevel panic" %
    #                    (self.videoFilePath, self.nDataLoaderThread, self.audioFilePath))
    #         subprocess.call(command, shell=True, stdout=None)

    def do(self, metadata_dict: dict) -> List[dict]:
        """
        Detect speaker in video.

        metadata_dict:
            Dict contains metadata.
        """
        sample = dict()
        sample['id'] = ['id']
        sample['channel'] = ['channel']
        sample['video_id'] = ['video_id']
        sample['video_path'] = [metadata_dict['video_path']]
        sample['demo'] = [True]

        samples = self.detector.process(
            sample,
            output_dir=self.output_dir,
            visual_output_dir=self.visual_output_dir,
            audio_output_dir=self.audio_output_dir,
            tmp_dir=self.tmp_dir,
            log_path=None,
        )

        _samples = []

        for _id, _c_id in zip(samples['id'],samples['chunk_visual_id']):
            if _id is None:
                continue
            _sample = dict()
            _sample['id'] = [_id]
            _sample['visual_path'] = [os.path.join(self.visual_output_dir, samples['channel'][0], _c_id) + '.mp4']
            _sample['visual_output_dir'] = [self.visual_output_dir]
            _sample['chunk_visual_id'] = [_c_id]
            _sample['visual_fps'] = [self.detector.V_FPS]
            _sample['visual_num_frames'] = [self.detector.V_FRAMES]
            _sample['audio_num_frames'] = [self.detector.A_FRAMES]
            _samples.append(_sample)

        assert _samples, self._logger.warning('No speaker is detected in video.')

        return _samples

