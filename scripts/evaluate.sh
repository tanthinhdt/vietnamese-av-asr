ROOT=$(dirname "$(dirname "$(readlink -fm "$0")")")
EXP=ViAVSP-LLM_v2.0
EVAL_SET=test

DATA_DIR=${ROOT}/data/processed/vasr/audio-visual/full
EXP_DIR=${ROOT}/models/${EXP}
MODEL_PATH=${EXP_DIR}/checkpoints/checkpoint_best.pt
LLM_PATH=vilm/vinallama-2.7b 

USE_BLEU=false
OUT_PATH=${EXP_DIR}/evaluations/${EVAL_SET}
SRC_DIR=${ROOT}/src

CUDA_VISIBLE_DEVICES=0 python -B ${SRC_DIR}/evaluate.py \
    --config-dir ${SRC_DIR}/configs/inference \
    --config-name decode \
        common.user_dir=${SRC_DIR} \
        dataset.gen_subset=${EVAL_SET} \
        override.data=${DATA_DIR} \
        override.label_dir=${DATA_DIR} \
        override.eval_bleu=${USE_BLEU} \
        override.llm_ckpt_path=${LLM_PATH} \
        common_eval.path=${MODEL_PATH} \
        common_eval.results_path=${OUT_PATH} \