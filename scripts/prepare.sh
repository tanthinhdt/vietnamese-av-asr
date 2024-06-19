#!/bin/bash

usage() {
  echo "Description:  Prepare environment when running on cloud platform kaggle or colab
  Usage: $0
    --platform <platform>         Cloud platform where to run project.
                                  2 available platforms 'kaggle' and 'colab'. Default: kaggle
  "
  exit 1
}

platform="kaggle"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --platform)
      if [[ -z "$2" ]];
      then
        echo "MISSING platform value"
        usage
      fi
      platform="$2"
      shift 2
    ;;
  esac
done


if [[ "$platform" != "kaggle" && "$platform" != "colab" ]];
then
  echo "Invalid platform."
  usage
fi

# Download and set up miniconda
MINICONDA_INSTALLER_SCRIPT=Miniconda3-py310_23.11.0-2-Linux-x86_64.sh
MINICONDA_PREFIX=/usr/local
wget https://repo.continuum.io/miniconda/$MINICONDA_INSTALLER_SCRIPT
chmod +x $MINICONDA_INSTALLER_SCRIPT
./$MINICONDA_INSTALLER_SCRIPT -b -f -p $MINICONDA_PREFIX

# Update version python
conda install --channel defaults conda python=3.10 --yes
conda update --channel defaults --all --yes

# Install ffmpeg to process media file
conda install -c conda-forge ffmpeg==7.0.1 --yes

if [[ "$platform" == 'kaggle' ]]
then
  conda install -c conda-forge git --yes
fi

# Install hf-transfer
pip install hf-transfer
env HF_HUB_ENABLE_HF_TRANSFER=1


# Install CocCoc tokenizer
pip install Cython
git clone https://github.com/coccoc/coccoc-tokenizer.git
cd coccoc-tokenizer
mkdir build
cd build
cmake -DBUILD_PYTHON=1 -DCMAKE_INSTALL_PREFIX=/usr/local ..
make install
cp /usr/local/lib/python3.10/site-packages/CocCocTokenizer-1.4-py3.10-linux-x86_64.egg/CocCocTokenizer.* /usr/local/lib/python3.10/site-packages
