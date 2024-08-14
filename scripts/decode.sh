ROOT=$(dirname "$(dirname "$(readlink -fm "$0")")")
EXP=ViAVSP-LLM_v2.0
EVAL_SET=test

DATA_PATH=${ROOT}/data/processed/vasr/audio-visual/full
MODEL_PATH=${ROOT}/models/${EXP}/checkpoints/checkpoint_best.pt
OUT_PATH=${ROOT}/evaluations/${EXP}/${EVAL_SET}

USE_BLEU=false
SRC_DIR=${ROOT}/src
LLM_PATH=vilm/vinallama-2.7b 

CUDA_VISIBLE_DEVICES=0 python -B ${SRC_DIR}/decode.py \
    --config-dir ${SRC_DIR}/configs/inference \
    --config-name decode \
        common.user_dir=${SRC_DIR} \
        dataset.gen_subset=${EVAL_SET} \
        override.data=${DATA_PATH} \
        override.label_dir=${DATA_PATH} \
        override.eval_bleu=${USE_BLEU} \
        override.llm_ckpt_path=${LLM_PATH} \
        common_eval.path=${MODEL_PATH} \
        common_eval.results_path=${OUT_PATH} \