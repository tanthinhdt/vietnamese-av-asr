#!/bin/bash

# Prepare dependencies for data collection
chmod +x src/data/scripts/prepare.sh
.src/data/scripts/prepare.sh

# Install packages for inference
pip install -r requirements.txt
