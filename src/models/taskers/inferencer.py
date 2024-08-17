import torch

from src.models.taskers import Checker, Normalizer, Splitter, MouthCropper, Embedder
from src.models.utils import get_logger, get_spent_time, clean_dirs
from src.models.utils.manifest import create_demo_manifest
from src.models.taskers.clustering import dump_feature, cluster_count, dump_label
from src.models.vsp_llm.vsp_llm_decode import produce_predictions

logger = get_logger('Inference', is_stream=True)


@get_spent_time(message='Inferencing time: ')
def infer(
        video_path: str,
        cfg=None,
        saved_cfg=None,
        llm_tokenizer=None,
        model: torch.nn.Module = None,
        extractor: torch.nn.Module = None,
        progress = None,
        **kwargs,

):
    checker = Checker(duration_threshold=180)
    normalizer = Normalizer()
    splitter = Splitter()
    mouth_cropper = MouthCropper()
    embedder = Embedder()

    logger.info('Start inferencing')

    logger.info(f"Check video")
    progress(progress=(1, 10), desc='Check video')
    checked_metadata = checker.do(video_path=video_path)

    if checked_metadata['has_v'] and checked_metadata['has_a']:
        modalities, short_modal = ["visual", "audio"], "av"
    elif checked_metadata['has_v']:
        modalities, short_modal = ["visual"], "v"
    else:
        modalities, short_modal = ["audio"], "a"

    logger.info(f"Normalize video")
    progress(progress=(2, 10), desc='Normalize video')
    normalized_metadata = normalizer.do(metadata_dict=checked_metadata, checker=checker)

    logger.info(f"Split into segments")
    progress(progress=(3, 10), desc=f"Split into second segments")
    samples = splitter.do(metadata_dict=normalized_metadata, time_interval=kwargs.get('time_interval', 3))

    logger.info(f"Crop mouth of speaker")
    progress(progress=(4, 10), desc='Crop mouth of speaker')
    samples = mouth_cropper.do(samples, need_to_crop=checked_metadata['has_v'])

    logger.info('Create manifest file')
    progress(progress=(5, 10), desc='Create manifest file')
    manifest_dir = create_demo_manifest(samples_dict=samples)

    logger.info('Extract features to cluster')
    progress(progress=(6, 10), desc='Extract features to cluster')
    dump_feature(
        extractor=extractor,
        tsv_dir=manifest_dir,
        split='test',
        nshard=1,
        rank=0,
        feat_dir=manifest_dir,
        user_dir='.',
        modalities=modalities,
    )

    logger.info("Assign units")
    progress(progress=(7, 10), desc='Assign units')
    dump_label(
        feat_dir=manifest_dir,
        split='test',
        km_path="src/models/checkpoints/kmean_model.km",
        lab_dir=manifest_dir,
    )

    logger.info("Cluster count")
    progress(progress=(8, 10), desc='Cluster count')
    cluster_count()

    logger.info("Predict transcripts")
    progress(progress=(9, 10), desc='Predict transcripts')
    produce_predictions(
        cfg=cfg,
        saved_cfg=saved_cfg,
        model=model,
        llm_tokenizer=llm_tokenizer,
        modalities=modalities,
    )

    logger.info('Embed transcript into video.')
    progress(progress=(10, 10), desc='Embed transcript into video.')
    _output_video_path = embedder.do(samples)

    logger.info("Clear fragments.")
    clean_dirs()

    logger.info('Inference DONE!')

    return _output_video_path