# Introduction
This project used to transcribe text of speaker in video.
The transcript is generated base on visual (lip movement) and audio signal from speaker.

# Docstring
Steps to clone repo, prepare environment and run inference demo.

## Clone Repository
Clone branch 'inference' from project repo.

```bash
git clone -b inference https://github.com/minhnv4099/vietnamese-av-asr.git
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

Download test demo [video file](https://drive.google.com/file/d/1kCgwpSPPAntC0HSCeCpOsDCQ_UmovijA/view?usp=share_link)
```bash
bash scripts/inference.sh <video-path>
```
##### Arguments:
`<video-path>`: Path to video file.

Result saved in [dir](decode)

## Video combine transcript
### You can see output video contains:
1. Scenes of speaker.
2. Transcripts corresponding scenes


### There are 2 version outputs:
1. Face of speaker, [here](decode/vsr/vi/output)
2. Origin scene, [here](decode/vsr/vi/output)

### Note:
1.  The output video is JUST INTUITIVE,
    means the transcript in video ASYNCHRONOUS with audio and visual. 
    Because of model's purpose.
2. The output video contains ONLY scenes of speaker




