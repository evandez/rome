"""Make knowns datasets to evaluate context mediation."""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import requests
from tqdm.auto import tqdm

KnownsDatasets = Dict[str, List[Dict]]

WINOVENTI_URL = "https://raw.githubusercontent.com/commonsense-exception/commonsense-exception/main/data/winoventi_bert_large_final.tsv"


def uncapitalize(string: str) -> str:
    """Uncapitalize the string."""
    return string[0].lower() + string[1:]


def count_occurrences(string: str, substring: str, lower: bool = True) -> int:
    """Count the occurrences of `substring` in `string."""
    if lower:
        string = string.lower()
        substring = substring.lower()
    occurrences = 0
    while substring in string:
        occurrences += 1
        string = string[string.index(substring) + 1 :]
    return occurrences


def make_counterfact(data_dir: Path) -> KnownsDatasets:
    """Convert CounterFact into knowns datasets for mediation."""
    import dsets

    counterfact = dsets.CounterFactDataset(str(data_dir))
    datasets: KnownsDatasets = defaultdict(list)
    known_id = 0
    for cf_sample in tqdm(counterfact, desc="make counterfact"):
        cf_requested_rewrite = cf_sample["requested_rewrite"]
        cf_subject = cf_requested_rewrite["subject"]
        cf_target_new = cf_requested_rewrite["target_new"]["str"]
        cf_target_true = cf_requested_rewrite["target_true"]["str"]
        cf_prompt = cf_requested_rewrite["prompt"].format(cf_subject)
        cf_generation_prompt = cf_sample["generation_prompts"][0]

        prompt = cf_generation_prompt
        if not prompt.startswith(cf_subject):
            prompt = uncapitalize(prompt)
        prompt = f"Suppose {prompt} {cf_target_new}. {cf_prompt}"

        for mediation, attribute, comparator in (
            ("med", cf_target_new, cf_target_true),
            ("umed", cf_target_true, cf_target_new),
        ):
            for key, subject, occurrence in (
                (f"{mediation}_subj_first", cf_subject, 0),
                (
                    f"{mediation}_subj_last",
                    cf_subject,
                    count_occurrences(prompt, cf_subject) - 1,
                ),
                (f"{mediation}_attr", cf_target_new, 0),
            ):
                sample = {
                    "known_id": known_id,
                    "subject": subject,
                    "prompt": prompt,
                    "attribute": attribute,
                    "occurrence": occurrence,
                    "comparator": comparator,
                }
                known_id += 1
                datasets[key].append(sample)
    return datasets


def make_winoventi(data_dir: Path) -> KnownsDatasets:
    """Convert WinoVenti into knowns datasets for mediation."""
    wv_file = data_dir / "winoventi.tsv"
    if not wv_file.exists():
        response = requests.get(WINOVENTI_URL)
        with wv_file.open("wb") as handle:
            handle.write(response.content)

    with wv_file.open("r") as handle:
        wv_samples = tuple(csv.DictReader(handle, delimiter="\t"))

    datasets: KnownsDatasets = defaultdict(list)
    known_id = 0
    for wv_sample in tqdm(wv_samples, desc="make winoventi"):
        wv_masked_prompt = wv_sample["masked_prompt"]
        wv_biased_word_context = wv_sample["biased_word_context"]
        wv_adv_word_context = wv_sample["adversarial_word_context"]
        wv_word = wv_sample["Word"]
        wv_target = wv_sample["target"]
        wv_incorrect = wv_sample["incorrect"]
        wv_type = wv_sample["test_type"]

        if int(wv_type) == 1:
            attribute = wv_biased_word_context
        else:
            attribute = wv_adv_word_context
        assert attribute in wv_masked_prompt

        prompt = wv_masked_prompt.replace("[MASK]", "").rstrip(". ")
        sample = {
            "known_id": known_id,
            "prompt": prompt,
            # Note: This is not a bug! The term "attribute" is overloaded, means
            # something different for ROME code than it does for WV.
            "attribute": wv_target,
            "comparator": wv_incorrect,
        }
        known_id += 1

        for key, subject, occurrence in (
            ("subj_first", wv_word, 0),
            ("subj_last", wv_word, count_occurrences(prompt, wv_word) - 1),
            ("attr", attribute, 0),
        ):
            datasets[key].append(
                {
                    "subject": subject,
                    "occurrence": occurrence,
                    **sample,
                }
            )

    return datasets


HANDLERS_FOR_DATASETS = {
    "counterfact": make_counterfact,
    "winoventi": make_winoventi,
}
DATASETS = tuple(sorted(HANDLERS_FOR_DATASETS))


def main():
    """Make the datasets."""
    parser = argparse.ArgumentParser(description="make mediation datasets")
    parser.add_argument("--data-dir", type=Path, help="path for raw data")
    parser.add_argument("--output-dir", type=Path, help="path for new KDs")
    parser.add_argument(
        "--datasets",
        choices=DATASETS,
        nargs="+",
        default=DATASETS,
        help="datasets to convert",
    )
    parser.add_argument(
        "--sample", type=int, default=1000, help="sample dataset down to this size"
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data"

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = data_dir / "mediation"

    for path in (data_dir, output_dir):
        path.mkdir(exist_ok=True, parents=True)

    for dataset in args.datasets:
        handler = HANDLERS_FOR_DATASETS[dataset]
        knowns_datasets = handler(data_dir)
        for key, knowns_dataset in knowns_datasets.items():
            knowns_dataset = random.sample(knowns_dataset, k=args.sample)
            knowns_dataset_file = output_dir / f"{dataset}_{key}.json"
            with knowns_dataset_file.open("w") as handle:
                json.dump(knowns_dataset, handle)


if __name__ == "__main__":
    main()
