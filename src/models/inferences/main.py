import os
import subprocess
import sys
sys.path.append(os.getcwd())
import argparse

from src.models.utils.manifest import create_demo_manifest
from src.models.taskers import Checker, Normalizer, Splitter, MouthCropper, Embedder
from src.models.utils import get_logger, get_spent_time

logger = get_logger('inference', is_stream=True)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Make parameter to infer video file.")

    parser.add_argument(
        'video_path',
        type=str,
        help="Path to video would to infer",
    )

    parser.add_argument(
        '--time-interval',
        type=int,
        required=False,
        default=3,
        help='Time interval to split',
    )

    return parser.parse_args()


@get_spent_time(message='Inferencing time: ')
def infer(args: argparse.Namespace):
    checker = Checker(duration_threshold=60)
    normalizer = Normalizer(checker=checker)
    mouth_cropper = MouthCropper()
    splitter = Splitter()
    embedder = Embedder()

    logger.info('Start inferencing')

    logger.info(f"Check video")
    checked_metadata = checker.do(video_path=args.video_path)

    if checked_metadata['has_v'] and checked_metadata['has_a']:
        modalities, short_modal = "visual audio", "av"
    elif checked_metadata['has_v']:
        modalities, short_modal = "visual", "v"
    else:
        modalities, short_modal = "audio", "a"

    logger.info(f"Normalize video")
    normalized_metadata = normalizer.do(metadata_dict=checked_metadata, checker=checker)

    logger.info(f"Split into segments")
    samples = splitter.do(metadata_dict=normalized_metadata, time_interval=args.time_interval)

    logger.info(f"Crop mouth of speaker")
    samples = mouth_cropper.do(samples, need_to_crop=checked_metadata['has_v'])

    logger.info('Create manifest file')
    manifest_dir = create_demo_manifest(samples_dict=samples)

    k_mean_path = "src/models/checkpoints/kmean_model.km"

    # dump hubert feature
    dump_h_f_cmd = (f"python src/models/features/dump_hubert_feature.py {manifest_dir} test "
                    f"src/models/checkpoints/large_vox_iter5.pt 12 1 0 "
                    f"{manifest_dir} --user_dir . --modalities {modalities}").split(' ')

    # dump label
    dump_l_cmd = f"python src/models/features/dump_km_label.py {manifest_dir} test {k_mean_path} 1 0 {manifest_dir}".split(' ')

    # rename dumped label file
    rename_l_cmd = f"for rank in $(seq 0 $((1 - 1))); do   cat {manifest_dir}/test_0_1.km; done > {manifest_dir}/test.km"

    # cluster count
    cl_count_cmd = "python src/models/features/cluster_counts.py".split(' ')

    # vsp_llm decode
    decode_cmd = [
        'bash',
        'src/models/scripts/decode.sh',
        '--demo',
        '--modal',
        short_modal,
    ]

    _cmd_dict = {
        'dump_h_f': dump_h_f_cmd,
        'dump_l': dump_l_cmd,
        'rename_l': rename_l_cmd,
        'cl_count': cl_count_cmd,
        #'decode': decode_cmd,
    }

    for k in _cmd_dict:
        logger.info(f"Doing step {k}")
        shell = False
        if k == 'rename_l':
            shell = True
        _return_code = subprocess.run(_cmd_dict[k], shell=shell, stdout=sys.stdout, capture_output=False).returncode

        if _return_code:
            logger.error(f'Error when {k}')
            exit(1)

    logger.info('Combine video and transcript.')
    _output_video_path = embedder.do(samples)

    logger.info("Clear fragments.")
    embedder.post_do()

    logger.info(f"Output path: '{_output_video_path}'")
    logger.info('Inference DONE!')


if __name__ == '__main__':
    p_args = get_args()
    infer(p_args)