import torch
from itertools import chain
from argparse import Namespace
from fairseq import checkpoint_utils, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from transformers import AutoTokenizer
from omegaconf import OmegaConf, DictConfig
from src.models.taskers.clustering import HubertFeatureReader


def load_feature_extractor(
        ckpt_path: str,
        layer,
        max_chunk=1600000,
        custom_utils=None
):
    extractor = HubertFeatureReader(ckpt_path, layer, max_chunk, custom_utils)
    return extractor


def load_ensemble_model(cfg_path: str):
    def main(main_cfg: DictConfig):
        if isinstance(main_cfg, Namespace):
            main_cfg = convert_namespace_to_omegaconf(main_cfg)
        assert main_cfg.common_eval.path is not None, "--path required for recognition!"
        llm_tokenizer = AutoTokenizer.from_pretrained(main_cfg.override.llm_ckpt_path)
        try:
            utils.import_user_module(main_cfg.common)
        except ImportError:
            pass

        use_cuda = torch.cuda.is_available()

        model_override_cfg = {
            'model': {
                'w2v_path': main_cfg.override.w2v_path,
                'llm_ckpt_path': main_cfg.override.llm_ckpt_path,
            }
        }
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            [main_cfg.common_eval.path, ],
            model_override_cfg, strict=False
        )

        models = [model.eval() for model in models]
        lms = [None]
        for model in chain(models, lms):
            if model is None:
                continue
            if use_cuda and not main_cfg.distributed_training.pipeline_model_parallel:
                model.encoder.cuda()
                model.avfeat_to_llm.cuda()
                model.half()
        model = models[0]

        return model, main_cfg, saved_cfg, llm_tokenizer

    cfg = OmegaConf.load(cfg_path)
    return main(main_cfg=cfg)


def load_state_dict_for_extractor(
        extractor: torch.nn.Module,
        model: torch.nn.Module
):
    encoder_state_dict = model.encoder.state_dict()
    rename_encoder_state_dict = dict()
    for k, v in encoder_state_dict.items():
        rename_encoder_state_dict[k[4:]] = v

    extractor.load_state_dict(rename_encoder_state_dict, strict=False)