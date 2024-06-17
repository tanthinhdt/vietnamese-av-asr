import logging
import os
import subprocess
import sys
sys.path.append(os.getcwd())
import argparse

from src.models.utils.demo import create_demo_mainfest
from src.models.taskers import Checker, Normalizer, ASDDetector, DemoCropper

logger = logging.getLogger('main')


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Make parameter to infer video file.")

    parser.add_argument(
        'video_path',
        type=str,
        help="Path to video would to infer",
    )

    parser.add_argument(
        '--demo',
        required=False,
        default=False,
        action='store_true',
        help="Demo. Use fitted k-mean model"
    )

    parser.add_argument(
        '--decode',
        required=False,
        default=False,
        action='store_true',
    )

    return parser.parse_args()


def infer(args: argparse.Namespace):
    checker = Checker()
    normalizer = Normalizer()
    detector = ASDDetector()
    cropper = DemoCropper()

    # check visual and audio
    checked_metadata = checker.do(video_path=args.video_path)

    # normalize video
    normalized_metadata = normalizer.do(metadata_dict=checked_metadata)

    # detect speaker
    samples = detector.do(metadata_dict=normalized_metadata)

    # crop mouth
    samples = cropper.do(samples)

    # create mainfest file
    logger.info('create mainfest file')
    create_demo_mainfest(samples_dict=samples)

    # dump hubert feature
    dump_h_f = ("python src/features/dump_hubert_feature.py src/models/dataset/vsr/vi/ test "
                "src/models/checkpoints/large_vox_iter5.pt 12 1 0 "
                "src/models/dataset/vsr/vi/ --user_dir .")
    dump_h_f_cmd = dump_h_f.split(' ')

    # learn k-means
    learn_k_means = "python src/features/learn_kmeans.py src/models/dataset/vsr/vi/ test 1 src/models/dataset/vsr/vi/km_model.km 200 --percent -1"
    learn_k_means_cmd = learn_k_means.split(' ')

    # dump label
    dump_l = "python src/features/dump_km_label.py src/models/dataset/vsr/vi/ test src/models/dataset/vsr/vi/km_model.km 1 0 src/models/dataset/vsr/vi/"
    dump_l_cmd = dump_l.split(' ')

    # rename dumped label file
    rename_l_cmd = "for rank in $(seq 0 $((1 - 1))); do   cat src/models/dataset/vsr/vi/test_0_1.km; done > src/models/dataset/vsr/vi/test.km"

    # cluster count
    cl_count = "python src/features/cluster_counts.py"
    cl_count_cmd = cl_count.split(' ')

    # vsp_llm decode
    decode_cmd = ['src/models/scripts/decode.sh']

    _cmds_dict = {
        'dump_h_f': dump_h_f_cmd,
        'learn_km': learn_k_means_cmd,
        'dump_l': dump_l_cmd,
        'rename_l': rename_l_cmd,
        'cl_count': cl_count_cmd,
        'decode': decode_cmd,
    }

    for k in _cmds_dict:
        logger.info(f"Doing step {k}")
        shell = False
        if (k == 'learn_km' and args.demo) or (k == 'decode' and not args.decode):
            continue
        if k == 'rename_l':
            shell = True
        _return_code = subprocess.run(_cmds_dict[k], shell=shell, stdout=sys.stdout, capture_output=False).returncode

        if _return_code:
            logger.warning(f'Error when {k}')
            exit()

    print('Inference DONE.')


if __name__ == '__main__':
    p_args = get_args()
    infer(args=p_args)
