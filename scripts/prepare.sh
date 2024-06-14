#!/bin/bash

# Prepare dependencies for data collection
chmod +x vietnamese-av-asr/src/data/scripts/prepare.sh
./vietnamese-av-asr/src/data/scripts/prepare.sh

# Install packages for inference
pip install -r vietnamese-av-asr/requirements.txt
