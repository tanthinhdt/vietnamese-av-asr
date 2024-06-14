# Introduction
This project used to transcribe text of speaker in video.
The transcript is generated base on visual (lip movement) and audio signal from speaker.

# Docstring
Steps to clone repo, prepare environment and run inference demo.
## Clone Repository
Clone branch 'inference' from project repo.
```bash
git clone -b inference https://github.com/tanthinhdt/vietnamese-av-asr.git
```

## Prepare environment
Go to project dir.
```bash
cd vietnamese-av-asr
```
Install packages, dependencies.
```bash
./scripts/prepare.sh
```

## Prepare checkpoints
- Download and move [VSP_LLM checkpoint](https://drive.google.com/file/d/1cQJ-RRZv9Qbl_4zyjZliQurcr_FwnB18/view?usp=share_link) to [checkpoints dir](src/models/checkpoints/).
- Download and move[ AV_HuBert checkpoint](https://drive.google.com/file/d/167-_DiLutzMZtDcnA69tdlp5KxwMmHxQ/view?usp=share_link) to [checkpoints dir](src/models/checkpoints/).
- Download [k-mean model](https://drive.google.com/file/d/1QRhlMRAclLgZ-sv8vQZBRMlG_jbvmArn/view?usp=share_link). 
  1. MUST rename it to 'km_model.km'.
  2. Move 'km_model.km' to [dataset dir](src/models/dataset/vsr/vi)

## Run inference
Download some video...

Download test demo [video file](https://drive.google.com/file/d/1kCgwpSPPAntC0HSCeCpOsDCQ_UmovijA/view?usp=share_link)
```bash
./inference.sh <video-path>
```
##### Arguments:
`<video-path>`: Path to video file.

Result saved in [dir](decode/vsr/vi)

## Video combine transcript
...




