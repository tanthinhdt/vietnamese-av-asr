usage() {
  echo "Usage: $0
    --demo                'demo' mode, otherwise 'decode' mode.
    --export-onnx         Export model to ONNX
    --use-onnx            Use onnx for inferencing instead of origin
  "
  exit 1
}

demo=False
modalities=["visual","audio"]
export_onnx=False
use_onnx=False

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --demo)
      demo=True
      shift 1
      ;;
    --modal)
      if [[ -z "$2" ]];then
        echo "Missing modal values."
        usage
      else
        if [[ "$2" == "a" ]]; then
          modalities=["audio"]
        elif [[ "$2" == "v" ]]; then
          modalities=["visual"]
        elif [[ "$2" == "av" ]] || [[ "$2" == "va" ]]; then
          modalities=["visual","audio"]
        else
          echo "Invalid modal $2"
          usage
        fi
        shift 2
      fi
      ;;
    --export-onnx)
      export_onnx=True
      shift 1
      ;;
    --use-onnx)
      use_onnx=True
      shift 1
      ;;
    *)
      echo "Unexpected flag $1"
      usage
  esac
done

if $export_onnx; then
  use_onnx=False
fi

if [[ -z "$modalities" ]]; then
  echo "Should explicitly select modal to avoid unexpected behaviours"
  usage
fi

LANG=vi

# set paths
ROOT=$(dirname "$(dirname "$(dirname "$(dirname "$(readlink -fn "$0")")")")")

SRC_DIR=${ROOT}/src
_MODEL_SRC=${SRC_DIR}/models

LLM_PATH="vilm/vinallama-2.7b"   # path to llama checkpoint

DATA_ROOT=${_MODEL_SRC}/dataset   # path to test dataset dir

MODEL_PATH=${_MODEL_SRC}/checkpoints/checkpoint_best.pt  # path to trained model
W2V_PATH=${_MODEL_SRC}/checkpoints/large_vox_iter5.pt  # path to trained model


OUT_PATH=${ROOT}/decode    # output path to save

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
        generation.beam=20 \
        generation.lenpen=0 \
        dataset.max_tokens=3000 \
        common_eval.path=${MODEL_PATH} \
        common_eval.results_path=${OUT_PATH}/${TASK}/${LANG} \
        override.data=${DATA_PATH} \
        override.label_dir=${DATA_PATH}\
        override.eval_bleu=${USE_BLEU} \
        override.llm_ckpt_path=${LLM_PATH} \
        override.w2v_path=${W2V_PATH} \
        override.demo=$demo \
        override.modalities=$modalities \
        override.export_onnx=$export_onnx \
        override.use_onnx=$use_onnx