import logging
import os
import subprocess
import sys
sys.path.append(os.getcwd())
import argparse

from src.models.utils.demo import create_demo_mainfest
from src.models.taskers import Checker, Normalizer, ASDetector, MouthCropper

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(lineno)d in <%(funcName)s> | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)

logger = logging.getLogger('inference_pipe')


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
    detector = ASDetector(n_process=2)
    cropper = MouthCropper()

    # check visual and audio
    logger.info(f"Check existing of visual and audio in video")
    checked_metadata = checker.do(video_path=args.video_path)

    # normalize video
    #normalized_metadata = normalizer.do(metadata_dict=checked_metadata)

    # detect speaker
    logger.info(f"Detect active speaker in video")
    samples = detector.do(metadata_dict=checked_metadata)

    # crop mouth
    logger.info(f"Crop mouth of speaker in video")
    samples = cropper.do(samples)

    # create mainfest file
    logger.info('create mainfest file')
    mainfest_dir = create_demo_mainfest(samples_dict=samples)

    # dump hubert feature
    dump_h_f_cmd = (f"python src/features/dump_hubert_feature.py {mainfest_dir}/ test "
                "src/models/checkpoints/large_vox_iter5.pt 12 1 0 "
                f"{mainfest_dir}/ --user_dir .").split(' ')

    # learn k-means
    learn_k_means_cmd = f"python src/features/learn_kmeans.py {mainfest_dir} test 1 {mainfest_dir}/km_model.km 200 --percent -1".split(' ')

    # dump label
    dump_l_cmd = f"python src/features/dump_km_label.py {mainfest_dir}/ test {mainfest_dir}/km_model.km 1 0 {mainfest_dir}/".split(' ')

    # rename dumped label file
    rename_l_cmd = f"for rank in $(seq 0 $((1 - 1))); do   cat {mainfest_dir}/test_0_1.km; done > {mainfest_dir}/test.km"

    # cluster count
    cl_count_cmd = "python src/features/cluster_counts.py".split(' ')

    # vsp_llm decode
    decode_cmd = ['src/models/scripts/decode.sh']

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
        if (k == 'learn_km' and args.demo) or (k == 'decode' and not args.decode):
            continue
        if k == 'rename_l':
            shell = True
        _return_code = subprocess.run(_cmd_dict[k], shell=shell, stdout=sys.stdout, capture_output=False).returncode

        if _return_code:
            logger.warning(f'Error when {k}')
            exit()

    print('Inference DONE.')


if __name__ == '__main__':
    p_args = get_args()
    infer(args=p_args)
