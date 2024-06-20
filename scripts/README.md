# Introduction
Inferencing is coded to be able to run in both local machine and cloud platform such as Google Colab, Kaggle,...
Each cloud one has corresponding way to install dependencies, packages.
1. Local machine
2. Cloud platform

# Preparation

## Local machine
### 1. Create conda environment
```bash
conda create -n vietnamese-av-asr python=3.9 -y
```

### 2. Install ffmpeg
```bash
conda install -c conda-forge ffmpeg==7.0.1 --yes
```

### 3. Install hf-transfer to speed up downloading from huggingface hub
```bash
pip install hf-transfer
env HF_HUB_ENABLE_HF_TRANSFER=1
```


### 4. Install CocCoc tokenizer
```bash
pip install Cython
git clone https://github.com/coccoc/coccoc-tokenizer.git
cd coccoc-tokenizer
mkdir build
cd build
cmake -DBUILD_PYTHON=1 -DCMAKE_INSTALL_PREFIX=/usr/local ..
make install
cp /usr/local/lib/python3.9/site-packages/CocCocTokenizer-1.4-py3.9-linux-x86_64.egg/CocCocTokenizer.* /path/to/env/site-packages
```
### 5. Install packages
```bash
pip install -r vietnamese-av-asr/requirements.txt
```
## Cloud platform
### 1. Install dependencies 
```bash
bash vietnamese-av-asr/scripts/prepare.sh --platform <platform>
```
Argument:

`<platform>:` which platform run project, `kaggle` or `colab`. Default `kaggle`
### 2. Install packages
```bash
pip install -r vietnamese-av-asr/requirements.txt
```