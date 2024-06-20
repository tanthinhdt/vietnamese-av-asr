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
```bash
python src/models/inferences/main.py 
    <video-path>
    [--n-cluster <n-cluster>]
    [--time-interval <time-interval>]
```
##### Arguments:
- `<video-path>`: Path to video file. It can be video/visual/audio clip. It is required duration not to be greater than **30** seconds, because of resource constraints.
- `--n-cluster`: Number of clusters when learn k-means. Default 25.
- `--time-interval`: Time interval to split. Default 3s.

## Video with embedded transcript
### Output video is located in [dir](results)
#### Note: 
The output video is **JUST INTUITIVE**, means the transcript in video ASYNCHRONOUS with both audio and visual. 
Because of model's purpose, **JUST** transcribe **WITHOUT** time stamp.