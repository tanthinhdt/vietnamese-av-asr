ROOT=$(dirname "$(dirname "$(readlink -fm "$0")")")
EXP=ViAVSP-LLM_v2.0

DATA_DIR=${ROOT}/data/processed/vasr/audio-visual/full
EXP_DIR=${ROOT}/models
OUT_PATH=${EXP_DIR}/${EXP}

SRC_DIR=${ROOT}/src
LLM_PATH=vilm/vinallama-2.7b
PRETRAINED_MODEL_PATH=${ROOT}/models/AV-Hubert/large_vox_iter5.pt

fairseq-hydra-train \
    --config-dir ${SRC_DIR}/configs/training \
    --config-name ${EXP} \
        common.user_dir=${SRC_DIR} \
        task.data=${DATA_DIR} \
        task.label_dir=${DATA_DIR} \
        task.llm_ckpt_path=${LLM_PATH} \
        model.w2v_path=${PRETRAINED_MODEL_PATH} \
        model.llm_ckpt_path=${LLM_PATH} \
        hydra.run.dir=${OUT_PATH} \