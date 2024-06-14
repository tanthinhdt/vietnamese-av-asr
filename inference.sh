#!/bin/bash

usage() {
    echo "Inference video file
    Usage: 
          $0 <video-path>
    "
    exit 1
}

video_path=''

if [[ -z "$1" ]]; then
  echo "Missing positional video-path argument"
  usage
else
  video_path=$1
fi

python src/models/inferences/main.py $video_path --decode --demo
