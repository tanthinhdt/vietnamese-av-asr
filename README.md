# ViAVSP-LLM (Vietnamese Audio-Visual Speech Processing incorporated with LLM)

This is the PyTorch code for [Vietnamese Automatic Speech Recognition Utilizing Auditory and Visual Data](docs/report.pdf). This code is developed on the code of [VSP-LLM](https://github.com/Sally-SH/VSP-LLM).

# Introduction

We propose ViAVSP-LLM—a novel framework that harnesses the powerful context modeling capabilities of large language models (LLMs) to advance Vietnamese audio-visual speech processing. By employing a self-supervised visual speech model, our approach maps input video directly into the latent space of an LLM, enabling a seamless integration of visual and linguistic data. To address the issue of redundant information in input frames, we introduce a deduplication technique that effectively reduces the embedded audio-visual features. Coupled with Low-Rank Adaptation (LoRA), this method allows ViAVSP-LLM to be trained in a computationally efficient manner, optimizing both performance and resource utilization.

![demo](docs/demo.gif)

# Results

| Model            | VASR Test WER (%) | VASR Test CER (%) | Config | Checkpoint |
|------------------|:-----------------:|:-----------------:|:------:|:----------:|
| ViAVSP-LLM (base)  | 17.28  | 10.56 | [config](src/configs/training/ViAVSP-LLM_v1.0.yaml) | [huggingface](https://huggingface.co/GSU24AI03-SU24AI21/ViVSP-LLM_v1.0/tree/main) |
| ViAVSP-LLM (final) | 12.03  | 7.2   | [config](src/configs/training/ViAVSP-LLM_v1.0.yaml) | [huggingface](https://huggingface.co/GSU24AI03-SU24AI21/ViVSP-LLM_v1.0/tree/main) |


# Demo

Try our ViAVSP-LLM [demo]() on HuggingFace.

# Installation
1. Create an environment with `python==3.9.19`
    ```
    conda create -n vasr python=3.9.19 -y
    conda activate vasr
    ```
2. Install `torch`, `torchvision` and `torchaudio`
    ```
    pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
    ```
3. Install `fairseq`
    ```
    cd src/libs
    pip install -e fairseq
    ```
4. Install other requirements
    ```
    cd ../..
    pip install -r requirements.txt
    ```


# Data

## Downloading
The VASR dataset is available [here](https://huggingface.co/datasets/tanthinhdt/vasr) for research use under strict ethical guidelines. To ensure the protection of speaker anonymity, prospective users are required to provide their contact information and formally agree to the stipulated terms and conditions before gaining access to the dataset. This process is not only a formality but a crucial step in upholding privacy standards and promoting the responsible and ethical use of sensitive data.

## Dataset layout
```
└── data
    |
    ├── raw
    |   |
    |   └── vasr
    |       |
    |       ├── audio
    |       |   |
    |       |   ├── 000
    |       |   |   |
    |       |   |   ├── 0000016.wav
    |       |   |   ├── ...
    |       |   |   └── <example_id>.wav
    |       |   |
    |       |   ├── ...
    |       |   └── <shard_id>
    |       |
    |       ├── visual
    |       |   |
    |       |   ├── 000
    |       |   |   |
    |       |   |   ├── 0000016.mp4
    |       |   |   ├── ...
    |       |   |   └── <example_id>.mp4
    |       |   |
    |       |   ├── ...
    |       |   └── <shard_id>
    |       |
    |       └── metadata.parquet
    |
    └── processed
        |
        └── vasr
            |
            ├── train.tsv               # List of audio and video path for training
            ├── train.wrd               # List of target label for training
            ├── train.cluster_counts    # List of clusters to deduplication in training
            ├── valid.tsv               # List of audio and video path for validation
            ├── valid.wrd               # List of target label for validation
            ├── valid.cluster_counts    # List of clusters to deduplication in validation
            ├── test.tsv                # List of audio and video path for testing
            ├── test.wrd                # List of target label for testing
            └── test.cluster_counts     # List of clusters to deduplication in testing
```

## Preprocessing
### 1. Create manifest for splits.
Run the following command to create manifest for training, validation and test splits.
```
python src/process_data.py \
    --process create_manifest \
    --data_dir data/raw/vasr \
    --split <split> \
    --frac <frac-of-split> \
    --output_dir data/processed/vasr \
```
* `split`: Split name to extract (train, valid, test).
* `frac`: Percent of the split. Enter `1` for entire split.

### 2. Extract audio-visual features using AV-HuBERT.
Run the following command to extract audio-visual features.
```
python src/process_data.py \
    --process dump_feature \
    --tsv_dir path/to/manifest/file.tsv \
    --split <split> \
    --nshard <num_shards> \
    --rank <rank> \
    --feat_dir path/to/output/feature/directory \
    --ckpt_path path/to/AV-Hubert/large_vox_iter5.pt \
    --layer 12 \
```
* `split`: Split name to extract (train, valid, test).
* `nshard`: Number of shards.
* `rank`: Which shard to process (from 0 to `nshard`)

### 3. Train K-Means model.
Run the following command to create manifest for training, validation and test splits.
```
python src/process_data.py \
    --process learn_kmeans \
    --feat_dir path/to/feature/directory \
    --split <split> \
    --nshard <nshard> \
    --km_path path/to/output/km_model.km \
    --n_clusters <n_clusters> \
    --percent <percent>
```
* `split`: Split name to extract (train, valid, test).
* `nshard`: Number of shards. Must be consistent with one at Step 2.
* `n_clusters`: Number of clusters inputted to K-Means.
* `percent`: Percent of the split to train K-Means.

### 4. Get pseudo labels from K-Means model.
Run the following command to create manifest for training, validation and test splits.
```
python src/process_data.py \
    --process dump_label \
    --feat_dir path/to/feature/directory \
    --split <split> \
    --km_path path/to/output/km_model.km \
    --nshard <nshard> \
    --rank <rank> \
    --lab_dir path/to/output/labels/directory
```
* `split`: Split name to extract (train, valid, test).
* `nshard`: Number of shards. Must be consistent with one at Step 2 and 3.
* `rank`: Which shard to process (from 0 to `nshard`)

### 5. Count similar frames.
Run the following command to create manifest for training, validation and test splits.
```
python src/process_data.py \
    --process count_clusters \
    --split <split> \
    --nshard <nshard> \
    --lab_dir path/to/output/labels/directory \
    --output_dir path/to/output/directory
```
* `split`: Split name to extract (train, valid, test).
* `nshard`: Number of shards. Must be consistent with one at Step 2 and 3.

# Pretrained Backbones

Use these pretrained backbones for training.
| Backbone | Checkpoint |
|----------|:----------:|
| AV-HuBERT Large (LSR3 + VoxCeleb2) | [link](http://facebookresearch.github.io/av_hubert) |
| VinaLLaMA | [link](https://huggingface.co/vilm/vinallama-2.7b)  |

# Training

Open the training script ([`scripts/train.sh`](https://github.com/Sally-SH/VSP-LLM/blob/main/scripts/train.sh)) and replace these variables:

```bash
# Experiment's name.
EXP=???

# Path to training dataset directory.
DATA_DIR=???

# Path to where experiments will be located.
EXP_DIR=???

# Path to downloaded pre-trained AV-HuBERT.
PRETRAINED_MODEL_PATH=???

# HuggingFace LLaMA repo ID or path to LLaMA checkpoint.
LLM_PATH=???
```

Run the training script:

```bash
$ bash scripts/train.sh
```

# Decoding

Open the decoding script ([`scripts/decode.sh`](https://github.com/Sally-SH/VSP-LLM/blob/main/scripts/decode.sh)) and replace these variables:

```bash
# Experiment's name.
EXP=???

# Evaluation set to be used.
EVAL_SET=???

# Path to evaluation dataset directory.
DATA_DIR=???

# Path to where experiments will be located.
EXP_DIR=???

# Path to the trained model.
MODEL_PATH=???

# HuggingFace LLaMA repo ID or path to LLaMA checkpoint.
LLM_PATH=???
```

Run the decoding script:

```bash
$ bash scripts/decode.sh
```
