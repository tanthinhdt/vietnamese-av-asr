# Introduction
Inferencing is coded to be able to run in both local machine and cloud platform such as Google Colab, Kaggle,...
Each one has corresponding way to install dependencies, packages.
1. Local machine
2. Cloud platform

# Preparation

## Local machine
### 1. Create conda environment
```bash
conda create -n vietnamese-av-asr python=3.10 -y
``` 

[//]: # (Make sure python's version is 3.10 in order to avoid unexpected errors)

[//]: # (Check version fo python)

[//]: # (```bash)

[//]: # (python --version #should be 3.10.*)

[//]: # (```)

[//]: # (Update python version if not matched)

[//]: # (```bash)

[//]: # (conda install --channel defaults conda python=3.10 --yes)

[//]: # (```)

### 2. Install hf-transfer to speed up downloading from huggingface hub
```bash
pip install hf-transfer
env HF_HUB_ENABLE_HF_TRANSFER=1
```

### 3. Install CocCoc tokenizer
```bash
pip install Cython
git clone https://github.com/coccoc/coccoc-tokenizer.git
cd coccoc-tokenizer
mkdir build
cd build
cmake -DBUILD_PYTHON=1 -DCMAKE_INSTALL_PREFIX=/usr/local ..
make install
cp /usr/local/lib/python3.10/site-packages/CocCocTokenizer-1.4-py3.10-linux-x86_64.egg/CocCocTokenizer.* /usr/local/lib/python3.10/site-packages
```
### 4. Install packages
```bash
pip install -r vietnamese-av-asr/requirements.txt
```
## Cloud platform
### 1. Install dependencies 
```bash
bash vietnamese-av-asr/scripts/prepare.sh
```
### 2. Install packages
```bash
pip install -r vietnamese-av-asr/requirements.txt
```