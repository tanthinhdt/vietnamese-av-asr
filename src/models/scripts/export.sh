#! /bin/bash

DIR=$(dirname "$(readlink -fn "$0")")

bash $DIR/decode.sh --modal av --demo --export-onnx