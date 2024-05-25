import os
import sys

sys.path.append(os.getcwd())

import glob
import argparse
import polars as pl
from tqdm import tqdm
from CocCocTokenizer import PyTokenizer
from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to data directory.",
    )
    parser.add_argument(
        "--channel-names-path",
        type=str,
        default=None,
        help="Path to file containing channel names.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """
    Main function.
    """
    final_stage = sorted(glob.glob(os.path.join(args.data_dir, "stage_*")))[-1]
    final_stage_dir = os.path.join(args.data_dir, final_stage)

    transcript_dir = os.path.join(args.data_dir, "transcripts")
    if not os.path.exists(transcript_dir):
        raise ValueError(f"Directory {transcript_dir} does not exist.")

    statistics_dir = os.path.join(args.data_dir, "statistics")
    os.makedirs(statistics_dir, exist_ok=True)

    if args.channel_names_path:
        with open(args.channel_names_path, "r") as f:
            channel_names = f.read().strip().split()
    else:
        channel_names = os.listdir(args.data_dir)

    tokenizer = PyTokenizer(load_nontone_data=True)

    dictionary = {}
    total_samples = 0
    total_duration = 0
    total_gibberish = 0
    print("\n" + "#" * 50 + " Doing statistics " + "#" * 50)
    for channel_name in tqdm(
        channel_names,
        desc="Processing channels",
        total=len(channel_names),
        unit="channel"
    ):
        print("-" * 20 + f" Processing {channel_name} " + "-" * 20)

        # Get dataset.
        print("Preparing dataset...")
        final_stage_path = os.path.join(final_stage_dir, channel_name + ".json")
        if not os.path.exists(final_stage_path):
            print(f"Channel {channel_name} does not exist.")
            continue
        dataset = load_dataset("json", data_files=final_stage_path, split="train")

        # Do statistics on dataset.
        print("Doing statistics on dataset...")
        for sample in tqdm(
            dataset,
            desc="Doing statistics on dataset",
            total=dataset.num_rows,
            unit="sample"
        ):
            transcript_path = os.path.join(transcript_dir, channel_name, sample["id"] + ".txt")
            with open(transcript_path, "r") as f:
                words = f.read().split()
            for word in words:
                dictionary[word] = dictionary.get(word, 0) + 1
                if len(tokenizer.word_tokenize(word, tokenize_option=2)) > 1:
                    total_gibberish += 1
            total_duration += sample["duration"]
        total_samples += dataset.num_rows
        print("-" * (13 + len(channel_name) + 2 * 20))

    # Save statistics.
    print("Saving statistics...")
    words_df = pl.DataFrame({
        "word": list(dictionary.keys()),
        "count": list(dictionary.values()),
    })
    words_df = words_df.sort("count", descending=True)
    words_df.write_csv(os.path.join(statistics_dir, "words.csv"))

    with open(os.path.join(statistics_dir, "statistics.txt"), 'w') as f:
        print(f"Number of samples: {total_samples}", file=f)
        print(f"Number of vocabularies: {len(dictionary)}", file=f)
        print(f"Total number of words: {sum(dictionary.values())}", file=f)
        print(f"Total duration: {total_duration}s", file=f)
        print(f"Total gibberish: {total_gibberish}", file=f)

    dataset.cleanup_cache_files()


if __name__ == "__main__":
    main(parse_args())
