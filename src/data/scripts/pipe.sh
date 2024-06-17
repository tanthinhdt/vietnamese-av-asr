#!/bin/bash

# Show usage
usage() {
    echo "Usage: $0 
        --url <url> 
        [--channel-name <channel-name>]
        [--cache-dir <cache-dir>]
        [--output-dir <output-dir>]
        [--overwrite]
    "
    exit 1
}

# Initialize variables
url=""
channel_name=""
cache_dir=""
output_dir=""
overwrite=""

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --url)
            url="$2"
            shift 2
            ;;
        --channel-name)
            channel_name="$2"
            shift 2
            ;;
        --cache-dir)
            cache_dir="$2"
            shift 2
            ;;
        --output-dir)
            output_dir="$2"
            shift 2
            ;;
        --overwrite)
            overwrite="--overwrite"
        --help)
            usage
            ;;
        *)
            echo "Unknown parameter passed: $1"
            usage
            ;;
    esac
done

if [ -z "$url" ]; then
    echo "Missing argument: --url <url> is required."
    usage
fi
if [ -z "$cache_dir" ]; then
    # echo "Missing argument: --cache-dir <cache_dir> is required."
    # usage
    cache_dir="data/external/"
fi
if [ -z "$output_dir" ]; then
    #echo "Missing argument: --output-dir <output_dir> is required."
    #usage
    output_dir="data/processed/"
fi
if [ -z "$channel_name" ]; then
    channel_name="batch_99999"
fi
if [ -z "$overwrite" ]; then
    overwrite="--overwrite"
fi

# Unable
exit 1

# Track url and upload to track hub.
python src/data/tasks/track.py --url $url --channel $channel_name $overwrite --demo

# Download video from recently tracked url.
python src/data/tasks/process.py --task download --channel $channel_name --output-dir $output_dir $overwrite --upload-to-hub

# Dectect speaker in recently downloaded video.
python src/data/tasks/process.py --task asd --channel $channel_name --output-dir $output_dir $overwrite --upload-to-hub

# Crop mouth of speaker in recently as-detected video.
python src/data/tasks/process.py --task crop --channel $channel_name --output-dir $output_dir $overwrite --upload-to-hub

# Detect vietnamese in recently processed sample
python src/data/tasks/process.py --task vndetect --channel $channel_name --output-dir $output_dir $overwrite --upload-to-hub

# Transcript recently processed sample
python src/data/tasks/process.py --task transcribe --channel $channel_name --output-dir $output_dir $overwrite --upload-to-hub