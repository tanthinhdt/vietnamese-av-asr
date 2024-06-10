import os
import argparse
import logging
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Create manifest for VASR dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing visual, audio and text data.'
    )
    parser.add_argument(
        '--split',
        type=str,
        required=True,
        help='Split to create manifest for.'
    )
    parser.add_argument(
        '--training-type',
        type=str,
        default='pretrain',
        help='Type of training data to create manifest for.'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save manifest.',
    )
    return parser.parse_args()


def create_manifest(
    split: str,
    split_df: pd.DataFrame,
    data_dir: str,
    output_dir: str,
) -> None:
    logging.info('Creating manifest...')

    manifest = []
    texts = []
    for i, sample in enumerate(split_df.itertuples()):
        logging.info(f'[{i+1}/{len(split_df)}] Processing {sample.id}')

        rel_visual_path = os.path.join(
            'visual',
            split + '_' + str(sample.shard).zfill(4),
            f'{sample.id}.mp4',
        )
        visual_path = os.path.join(data_dir, rel_visual_path)
        if not os.path.exists(visual_path):
            logging.error(f'File {visual_path} does not exist.')
            continue

        rel_audio_path = os.path.join(
            'audio',
            split + '_' + str(sample.shard).zfill(4),
            f'{sample.id}.wav',
        )
        audio_path = os.path.join(data_dir, rel_audio_path)
        if not os.path.exists(audio_path):
            logging.error(f'File {audio_path} does not exist.')
            continue

        manifest.append(
            '\t'.join([
                sample.id,
                rel_visual_path,
                rel_audio_path,
                str(sample.video_num_frames),
                str(sample.audio_num_frames),
            ])
        )
        texts.append(sample.transcript)

    with open(os.path.join(output_dir, f'{split}.tsv'), 'w') as f:
        f.write(data_dir + '\n')
        f.write('\n'.join(manifest) + '\n')
    with open(os.path.join(output_dir, f'{split}.wrd'), 'w') as f:
        f.write('\n'.join(texts) + '\n')

    logging.info(f'{split} set have {len(texts)} sample')


def main(args: argparse.Namespace) -> None:
    if not os.path.exists(args.data_dir):
        logging.error(f'Directory {args.data_dir} does not exist.')
        return
    metadata_path = os.path.join(
        args.data_dir,
        args.training_type,
        f'{args.split}_completed.parquet',
    )
    if not os.path.exists(metadata_path):
        logging.error(f'File {metadata_path} does not exist.')
        return
    os.makedirs(args.output_dir, exist_ok=True)

    split_df = pd.read_parquet(metadata_path)
    logging.info(f'Found {len(split_df)} ids.')

    create_manifest(
        split=args.split,
        split_df=split_df,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    args = get_args()
    main(args=args)
