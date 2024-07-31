from pathlib import Path
from datasets import Dataset, Audio


def load_test_set(test_manifest_file: str, test_ref_file: str) -> Dataset:
    with open(test_manifest_file) as f:
        root_dir = Path(f.readline().strip())
        audio_files = [str(root_dir / line.split()[2]) for line in f.readlines()]
    with open(test_ref_file) as f:
        refs = [line.strip() for line in f.readlines()]
    test_set = Dataset.from_dict({"audio": audio_files, "ref": refs})
    return test_set.cast_column("audio", Audio())
