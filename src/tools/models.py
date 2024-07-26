import torch
import logging
from time import time
from typing import Union
import onnxruntime as ort
from transformers import (
    FeatureExtractionMixin,
    AutoModel,
    Pipeline,
    pipeline,
)
from transformers.pipelines import PIPELINE_REGISTRY
from configs import ModelConfig, InferenceConfig
from models import (
    AVHubertConfig, AVHubertFeatureExtractor, AVHubertModel,
    AVSPLLMConfig, AVSPLLMFeatureExtractor, AVSPLLMModel,
)
from pipelines import AutomaticSpeechRecognitionPipeline
from utils import MODELS, draw_text_on_image


def load_model(
    model_config: ModelConfig,
    do_train: bool = False,
) -> tuple:
    """
    """
    if do_train:
        if model_config.arch == "av_hubert":
            config_class = AVHubertConfig
            processor_class = AVHubertFeatureExtractor
            model_class = AVHubertModel
        elif model_config.arch == "avsp_llm":
            config_class = AVSPLLMConfig
            processor_class = AVSPLLMFeatureExtractor
            model_class = AVSPLLMModel
        else:
            logging.error(f"Model {model_config.arch} is not supported")
            exit(1)

        config_class.register_for_auto_class()
        processor_class.register_for_auto_class("AutoFeatureExtractor")
        model_class.register_for_auto_class("AutoModel")
        logging.info(f"{model_config.arch} classes registered")

        config = config_class(**vars(model_config))
        processor = processor_class(config=config)
        model = model_class(config=config)

        return config, processor, model

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

    return model.config, processor, model


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


def get_predictions(
    inputs: torch.Tensor,
    model: Union[ort.InferenceSession, AutoModel],
    id2gloss: dict,
    k: int = 3,
) -> Predictions:
    """
    Get the top-k predictions.
    Parameters
    ----------
    inputs : torch.Tensor
        Model inputs (Time, Height, Width, Channels).
    model : Union[ort.InferenceSession, AutoModel]
        Model to get predictions from.
    id2gloss : dict
        Mapping of class indices to glosses.
    k : int, optional
        Number of predictions to return, by default 3.
    Returns
    -------
    tuple
        List of top-k predictions and inference time.
    """
    if inputs is None:
        return Predictions()

    # Get logits
    start_time = time()
    if isinstance(model, ort.InferenceSession):
        inputs = inputs.cpu().numpy()
        logits = torch.from_numpy(model.run(None, {"pixel_values": inputs})[0])
    else:
        logits = model(inputs.to(model.device)).logits
    inference_time = time() - start_time

    # Get top-3 predictions
    topk_scores, topk_indices = torch.topk(logits, k, dim=1)
    topk_scores = torch.nn.functional.softmax(topk_scores, dim=1).squeeze().detach().numpy()
    topk_indices = topk_indices.squeeze().detach().numpy()
    predictions = [
        {
            'gloss': id2gloss[str(topk_indices[i])],
            'score': topk_scores[i],
        }
        for i in range(k)
    ]

    return Predictions(predictions=predictions, inference_time=inference_time)


def load_pipeline(
    model_config: ModelConfig,
    inference_config: InferenceConfig = None,
) -> Pipeline:
    """
    """
    assert model_config.arch in MODELS, f"Model {model_config.arch} is not supported"

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


def get_dummy_input(
    arch: str,
    processor: Union[FeatureExtractionMixin],
    batch_size: int = 1,
) -> tuple:
    """
    Get the input shape for the model.

    Parameters
    ----------
    processor : Union[FeatureExtractionMixin]
        Model processor.
    batch_size : int, optional
        Batch size, by default 1.

    Returns
    -------
    tuple
        Input shape.
    """
    assert arch in MODELS, f"Model {arch} is not supported"

    if arch == "avsp_llm":
        source = {
            "audio": torch.randn(
                batch_size,
                processor.audio_feat_dim,
                processor.num_frames,
            ),
            "video": torch.randn(
                batch_size,
                processor.num_channels,
                processor.num_frames,
                processor.height,
                processor.width,
            ),
            "cluster_counts": [torch.randn(processor.cluster_counts_dim)],
            "text": torch.randn(batch_size, processor.text_feat_dim),
        }

        net_input = {
            "source": source,
            "padding_mask": torch.zeros(batch_size, processor.num_frames),
            "text_attn_mask": torch.zeros(batch_size, processor.text_feat_dim),
        }

        return net_input
