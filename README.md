# Introduction
This project used to transcribe text of speaker in video.
The transcript is generated base on visual (lip movement) and audio signal from speaker.

# Docstring
Steps to clone repo, prepare environment and run inference demo.

## Clone Repository
Clone branch 'inference' from project repo. Use accessible token to clone.  
```bash
git clone -b inference https://github.com/minhnv4099/vietnamese-av-asr.git
```

## Prepare environment
Read [prepare environment.](scripts/README.md) 

## Prepare checkpoints
- Download and move [VSP_LLM checkpoint](https://drive.google.com/file/d/1dKXL0cYrTCqkfuIBuo1avlAlGwIPQhP1/view?usp=sharing) to [checkpoints dir](src/models/checkpoints/).

[//]: # (## Go to project dir)

[//]: # (```bash)

[//]: # (cd vietnamese-av-asr/)

[//]: # (```)

[//]: # ()
[//]: # (## Run inference)

[//]: # (```bash)

[//]: # (python src/models/inferences/main.py )

[//]: # (    <video-path>)

[//]: # (```)

[//]: # (##### Arguments:)

[//]: # (- `<video-path>`: Path to video file. It can be video/visual/audio clip. It is required duration not to be greater than **30** seconds, because of resource constraints.)

[//]: # ()
[//]: # (## Video with embedded transcript)

[//]: # (### Output video and transcript are located in `result/` dir)

[//]: # (#### Note: )

[//]: # (The output video is **JUST INTUITIVE**, means the transcript in video ASYNCHRONOUS with both audio and visual. )

[//]: # (Because of model's purpose, **JUST** transcribe **WITHOUT** time stamp.)

[//]: # ()
[//]: # (## Deploy in [Gradio]&#40;https://huggingface.co/spaces/nguyenminh4099/Demo&#41; )