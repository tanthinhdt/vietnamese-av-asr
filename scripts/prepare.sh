#!/bin/bash


# Download and set up miniconda
MINICONDA_INSTALLER_SCRIPT=Miniconda3-py310_23.11.0-2-Linux-x86_64.sh
MINICONDA_PREFIX=/usr/local
wget https://repo.continuum.io/miniconda/$MINICONDA_INSTALLER_SCRIPT
chmod +x $MINICONDA_INSTALLER_SCRIPT
./$MINICONDA_INSTALLER_SCRIPT -b -f -p $MINICONDA_PREFIX


conda install --channel defaults conda python=3.10 --yes
conda update --channel defaults --all --yes

# Kaggle
conda install git --yes
conda install libffi==3.3 --yes


# Install hf-transfer
pip install hf-transfer
env HF_HUB_ENABLE_HF_TRANSFER=1


# Install Coccoc tokenizer
pip install Cython
git clone https://github.com/coccoc/coccoc-tokenizer.git
cd coccoc-tokenizer
mkdir build
cd build
cmake -DBUILD_PYTHON=1 -DCMAKE_INSTALL_PREFIX=/usr/local ..
make install
cp /usr/local/lib/python3.10/site-packages/CocCocTokenizer-1.4-py3.10-linux-x86_64.egg/CocCocTokenizer.* /usr/local/lib/python3.10/site-packages

# Install packages
pip install -r vietnamese-av-asr/requirements.txt

