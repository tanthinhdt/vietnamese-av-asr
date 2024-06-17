#! /bin/bash
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

LANG=vi    # language direction (e.g 'en' for VSR task / 'en-es' for En to Es VST task)

# set paths
ROOT=$(dirname "$(dirname "$(dirname "$(dirname "$(readlink -fn "$0")")")")")

SRC_DIR=${ROOT}/src
_MODEL_SRC=${SRC_DIR}/models

LLM_PATH="vilm/vinallama-2.7b"   # path to llama checkpoint

DATA_ROOT=${_MODEL_SRC}/dataset   # path to test dataset dir

MODEL_PATH=${_MODEL_SRC}/checkpoints/checkpoint_best.pt  # path to trained model
W2V_PATH=${_MODEL_SRC}/checkpoints/large_vox_iter5.pt  # path to trained model


OUT_PATH=${ROOT}/decode    # output path to save

# fix variables based on langauge
if [[ $LANG == *"-"* ]] ; then
    TASK="vst"
    IFS='-' read -r SRC TGT <<< ${LANG}
    USE_BLEU=true
    DATA_PATH=${DATA_ROOT}/${TASK}/${SRC}/${TGT}

else
    TASK="vsr"
    TGT=${LANG}
    USE_BLEU=false
    DATA_PATH=${DATA_ROOT}/${TASK}/${LANG}
fi

# start decoding
export PYTHONPATH="${ROOT}/fairseq:$PYTHONPATH"
export HYDRA_FULL_ERROR=1
CUDA_VISIBLE_DEVICES=0 python -B ${_MODEL_SRC}/vsp_llm/vsp_llm_decode.py \
    --config-dir ${_MODEL_SRC}/configs \
    --config-name s2s_decode \
        common.user_dir=${SRC_DIR} \
        dataset.gen_subset=test \
        override.data=${DATA_PATH} \
        override.label_dir=${DATA_PATH}\
        generation.beam=20 \
        generation.lenpen=0 \
        dataset.max_tokens=3000 \
        override.eval_bleu=${USE_BLEU} \
        override.llm_ckpt_path=${LLM_PATH} \
        override.w2v_path=${W2V_PATH} \
        common_eval.path=${MODEL_PATH} \
        common_eval.results_path=${OUT_PATH}/${TASK}/${LANG}