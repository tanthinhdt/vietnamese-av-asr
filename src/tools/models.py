import logging
from transformers import (
    FeatureExtractionMixin,
    AutoModel,
    Pipeline,
    pipeline,
)
from transformers.pipelines import PIPELINE_REGISTRY
from configs import ModelConfig, InferenceConfig
from modules import (
    AVSPLLMConfig,
    AVSPLLMFeatureExtractor,
    AVSPLLMModel,
)
from pipelines import AutomaticSpeechRecognitionPipeline


def load_model(
    model_config: ModelConfig,
    do_train: bool = False,
) -> tuple:
    """
    """
    if do_train:
        config_class = AVSPLLMConfig
        processor_class = AVSPLLMFeatureExtractor
        model_class = AVSPLLMModel

        config_class.register_for_auto_class()
        processor_class.register_for_auto_class("AutoFeatureExtractor")
        model_class.register_for_auto_class("AutoModel")
        logging.info("Classes registered")

        config = config_class(**vars(model_config))
        processor = processor_class(config=config)
        model = model_class(config=config)

        return processor, model

    processor = FeatureExtractionMixin.from_pretrained(
        model_config.pretrained,
        trust_remote_code=True,
        cache_dir="models/huggingface",
    )
    model = AutoModel.from_pretrained(
        model_config.pretrained,
        trust_remote_code=True,
        cache_dir="models/huggingface",
    )
    model.eval()

    return processor, model


def load_pipeline(
    model_config: ModelConfig,
    inference_config: InferenceConfig = None,
) -> Pipeline:
    """
    """
    if inference_config is None:
        processor, model = load_model(model_config)
        PIPELINE_REGISTRY.register_pipeline(
            "automatic-speech-recognition",
            pipeline_class=AutomaticSpeechRecognitionPipeline,
            pt_model=AutoModel,
            default={"pt": ("tanthinhdt/ViAVSP_LLM_v2_0", "main")},
            type="multimodal",
        )
        return AutomaticSpeechRecognitionPipeline(
            model=model,
            feature_extractor=processor,
            device=inference_config.device,
            model_kwargs={
                "cache_dir": inference_config.cache_dir,
            },
            assistant_model=inference_config.assistant_model,
        )

    return pipeline(
        "automatic-speech-recognition",
        model=model_config.pretrained,
        feature_extractor=model_config.pretrained,
        device=inference_config.device,
        model_kwargs={
            "cache_dir": inference_config.cache_dir,
        },
        assistant_model=inference_config.assistant_model,
        trust_remote_code=True,
    )
