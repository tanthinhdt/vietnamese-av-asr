# Introduction
This project used to transcribe text of speaker in video.
The transcript is generated base on visual (lip movement) and audio signal from speaker.

# Docstring
Steps to clone repo, prepare environment and run inference demo.

## Clone Repository
Clone branch 'inference' from project repo. Use accessible token to clone.  
```bash
git clone -b inference <repo-link>
```

## Prepare environment
Read [prepare environment.](scripts/README.md) 

## Prepare checkpoints
- Download and move [VSP_LLM checkpoint](https://drive.google.com/file/d/1cQJ-RRZv9Qbl_4zyjZliQurcr_FwnB18/view?usp=share_link) to [checkpoints dir](src/models/checkpoints/).
- Download and move[ AV_HuBert checkpoint](https://drive.google.com/file/d/167-_DiLutzMZtDcnA69tdlp5KxwMmHxQ/view?usp=share_link) to [checkpoints dir](src/models/checkpoints/).

## Go to project dir
```bash
cd vietnamese-av-asr/
```

## Run inference
Download some video...

Download test demo [video file](https://drive.google.com/file/d/1Qgar8SXyfPSeg7O69VtBNyUCZuSdP7LQ/view?usp=share_link)
```bash
bash scripts/inference.sh 
    <video-path>
    [--decode]
    [--clear-fragments]
    [--n-cluster <n-cluster>]
```
##### Arguments:
- `<video-path>`: Path to video file
- `--clear-fragments`: Clear intermediate results generated during inferencing progress.
- `--n-cluster`: Number of clusters when learn k-means.

## Video combines transcript
### Output video contains:
1. Scenes of speaker.
2. Transcripts corresponding scenes


### There are 2 version outputs:
1. Facial scene, [here](results)
2. Original scene, [here](results)

### Note:
1.  The output video is **JUST INTUITIVE**,
    means the transcript in video ASYNCHRONOUS with both audio and visual. 
    Because of model's purpose, **JUST** transcribe **WITHOUT** time stamp.
2. The output video contains **ONLY** scene of **SPEAKER**




