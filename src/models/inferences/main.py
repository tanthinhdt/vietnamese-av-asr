import os
import subprocess
import sys
sys.path.append(os.getcwd())
import argparse

from src.models.utils.mainfest import create_demo_mainfest
from src.models.taskers import Checker, DemoASDetector, DemoCropper, Combiner, Normalizer
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
        help='Interval to split'
    )

    parser.add_argument(
        '--n-cluster',
        type=int,
        required=False,
        default=20,
        help='N-cluster when clustering hubert features.'
    )

    parser.add_argument(
        '--decode',
        required=False,
        default=False,
        action='store_true',
        help='decode phase.'
    )

    parser.add_argument(
        '--clear-fragments',
        required=False,
        default=False,
        action='store_true',
        help='Clear fragments(intermediate results made by inferencing progress).'
    )

    return parser.parse_args()


@get_spent_time(message='Inferencing time in second: ')
def infer(args: argparse.Namespace):
    checker = Checker()
    normalizer = Normalizer()
    detector = DemoASDetector(time_interval=args.time_interval)
    cropper = DemoCropper()
    combiner = Combiner()

    logger.info('Start inferencing')

    # check visual and audio
    logger.info(f"Check video")
    checked_metadata = checker.do(video_path=args.video_path)

    logger.info(f"Normalize video")
    normalized_metadata = normalizer.do(metadata_dict=checked_metadata)

    # detect speaker
    logger.info(f"Detect active speaker in video")
    samples = detector.do(metadata_dict=normalized_metadata, clear_fragments=args.clear_fragments)

    # crop mouth
    logger.info(f"Crop mouth of speaker in video")
    samples = cropper.do(samples)

    # create mainfest file
    logger.info('Create mainfest file')
    mainfest_dir = create_demo_mainfest(samples_dict=samples)

    # dump hubert feature
    dump_h_f_cmd = (f"python src/features/dump_hubert_feature.py {mainfest_dir} test "
                "src/models/checkpoints/large_vox_iter5.pt 12 1 0 "
                f"{mainfest_dir} --user_dir .").split(' ')

    # learn k-means
    learn_k_means_cmd = (f"python src/features/learn_kmeans.py "
                         f"{mainfest_dir} test 1 {mainfest_dir}/km_model.km "
                         f"{args.n_cluster} --percent -1").split(' ')

    # dump label
    dump_l_cmd = f"python src/features/dump_km_label.py {mainfest_dir} test {mainfest_dir}/km_model.km 1 0 {mainfest_dir}".split(' ')

    # rename dumped label file
    rename_l_cmd = f"for rank in $(seq 0 $((1 - 1))); do   cat {mainfest_dir}/test_0_1.km; done > {mainfest_dir}/test.km"

    # cluster count
    cl_count_cmd = "python src/features/cluster_counts.py".split(' ')

    # vsp_llm decode
    decode_cmd = ['src/models/scripts/decode.sh', '--demo']

    _cmd_dict = {
        'dump_h_f': dump_h_f_cmd,
        'learn_km': learn_k_means_cmd,
        'dump_l': dump_l_cmd,
        'rename_l': rename_l_cmd,
        'cl_count': cl_count_cmd,
        'decode': decode_cmd,
    }

    for k in _cmd_dict:
        logger.info(f"Doing step {k}")
        shell = False
        if k == 'decode' and not args.decode:
            continue
        if k == 'rename_l':
            shell = True
        _return_code = subprocess.run(_cmd_dict[k], shell=shell, stdout=sys.stdout, capture_output=False).returncode

        if _return_code:
            logger.error(f'Error when {k}')
            exit()

    logger.info('Combine video and transcript.')
    _output_videos = combiner.do()

    if args.clear_fragments:
        logger.info('Clear fragments')
        combiner.post_do(clear_framents=args.clear_fragments)

    logger.info(f"2 output videos: '{_output_videos[0]}', '{_output_videos[1]}'")
    logger.info('Inference DONE!')


if __name__ == '__main__':
    p_args = get_args()
    infer(p_args)
