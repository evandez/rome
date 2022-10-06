"""Compute tokenization stats for reference."""
import argparse
import json
from collections import Counter
from pathlib import Path

import dsets

import transformers


def main():
    """Compute token statistics over knowns dataset."""
    parser = argparse.ArgumentParser(description="compute token stats")
    parser.add_argument("--model", default="gpt2", help="model tokenizer to use")
    parser.add_argument("--data-file", type=Path, help="json file containing knowns")
    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    if args.data_file is not None:
        with args.data_file.open("r") as handle:
            knowns = json.load(handle)
    else:
        knowns = dsets.KnownsDataset(Path(__file__).parent.parent / "data")
    subjects = [known["attribute"].strip() for known in knowns]
    tokenized = tokenizer(subjects, add_special_tokens=False)
    lengths = [len(tokens) for tokens in tokenized.input_ids]
    print("attribute token lengths", Counter(lengths))


if __name__ == "__main__":
    main()
