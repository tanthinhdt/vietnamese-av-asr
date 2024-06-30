#!/bin/bash

usage() {
    echo "Inference video file
Usage:
  $0
    <video-path>                    Path to video.
    [--clear-fragments]             Clear intermediate result generated during inferencing progress.
    [--n-cluster]                   Number of cluster when learn k-means. Default 100
    "
    exit 1
}

clear_fragments=""
n_cluster=25
video_path=''

if [[ -z "$1" ]]; then
  echo "Missing positional video-path argument"
  usage
else
  video_path=$1
fi

while [[ "$#" -gt 1 ]]; do
  case "$2" in
    --clear-fragments)
      clear_fragments="--clear-fragments"
      shift 1
      ;;
    --n-cluster)
      n_cluster="$3"
      shift 2
      ;;
    --help)
      usage
      ;;
    *)
      echo "Got unexpected argument '$2'"
      usage
      ;;
  esac
done

python src/models/inferences/main.py $video_path --n-cluster $n_cluster --decode --clear-fragments