import torch
import logging
from transformers import (
    FeatureExtractionMixin,
    AutoModel,
    Pipeline,
    pipeline,
)
from transformers.pipelines import PIPELINE_REGISTRY
from configs import ModelConfig, InferenceConfig
from models import (
    AVSPLLMConfig,
    AVSPLLMFeatureExtractor,
    AVSPLLMModel,
)
from pipelines import AutomaticSpeechRecognitionPipeline
from utils import draw_text_on_image


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
        logging.info(f"{model_config.arch} classes registered")

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


class Predictions:
    def __init__(
        self,
        predictions: list[dict] = None,
        inference_time: float = 0,
        start_time: float = 0,
        end_time: float = 0,
    ) -> None:
        self.predictions = predictions
        self.inference_time = inference_time
        self.start_time = start_time
        self.end_time = end_time

    def visualize(
        self,
        frame: torch.Tensor,
        position: tuple = (20, 100),
        prefix: str = "Predictions",
        color: tuple = (0, 0, 255),
    ) -> None:
        text = prefix + ": " + self.get_pred_message()
        return draw_text_on_image(
            image=frame,
            text=text,
            position=position,
            color=color,
            font_size=20,
        )

    def get_pred_message(self) -> str:
        if not any((
            self.start_time,
            self.end_time,
            self.inference_time,
            self.predictions
        )):
            return ""

        return ', '.join(
            [
                f"{pred['gloss']} ({pred['score']*100:.2f}%)"
                for pred in self.predictions
            ]
        )

    def __str__(self) -> str:
        if not any((
            self.start_time,
            self.end_time,
            self.inference_time,
            self.predictions
        )):
            return ""

        predictions = self.get_pred_message()
        message = "Sample start: {:.2f}s - end: {:.2f}s | Runtime: {:.2f}s | Predictions: {}"
        return message.format(self.start_time, self.end_time, self.inference_time, predictions)

    def merge_results(self, results: dict = None) -> dict:
        if results is None:
            results = {
                "start_time": [],
                "end_time": [],
                "inference_time": [],
                "prediction": [],
            }
        results["start_time"].append(self.start_time)
        results["end_time"].append(self.end_time)
        results["inference_time"].append(self.inference_time)
        results["prediction"].append(self.predictions)
        return results


def load_pipeline(
    model_config: ModelConfig,
    inference_config: InferenceConfig = None,
) -> Pipeline:
    """
    """
    if inference_config is None:
        _, processor, model = load_model(model_config)
        PIPELINE_REGISTRY.register_pipeline(
            "automatic-speech-recognition",
            pipeline_class=AutomaticSpeechRecognitionPipeline,
            pt_model=AutoModel,
            default={"pt": ("tanthinhdt/ViAVSP-LLM_v2.0", "main")},
            type="multimodal",
        )
        return AutomaticSpeechRecognitionPipeline(
            model=model,
            feature_extractor=processor,
        )

    return pipeline(
        "automatic-speech-recognition",
        model=model_config.pretrained,
        image_processor=model_config.pretrained,
        device=inference_config.device,
        model_kwargs={
            "cache_dir": inference_config.cache_dir,
        },
        trust_remote_code=True,
        use_onnx=inference_config.use_onnx,
    )
