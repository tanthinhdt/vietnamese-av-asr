#! /bin/bash


# Download and set up miniconda
MINICONDA_INSTALLER_SCRIPT=Miniconda3-py39_23.11.0-2-Linux-x86_64.sh
MINICONDA_PREFIX=/usr/local
wget https://repo.continuum.io/miniconda/$MINICONDA_INSTALLER_SCRIPT
chmod +x $MINICONDA_INSTALLER_SCRIPT
./$MINICONDA_INSTALLER_SCRIPT -b -f -p $MINICONDA_PREFIX


# Test and upgrade conda
which conda # should return /usr/local/bin/conda
conda --version # should return 23.11.0
python --version # should return 3.9.18

conda install --channel defaults conda python=3.9 --yes
conda update --channel defaults --all --yes


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
cp /usr/local/lib/python3.9/site-packages/CocCocTokenizer-1.4-py3.9-linux-x86_64.egg/CocCocTokenizer.* /usr/local/lib/python3.9/site-packages
# Test coccoc tokenizer
conda list | grep coccoctokenizer   # should show coccoctokenizer 1.4


# Install requirements
cd /content/vietnamese-av-asr/
pip install -r ./src/data/databases/requirements.txt
pip install -U datasets
pip install fsspec
cd /content/vietnamese-av-asr/