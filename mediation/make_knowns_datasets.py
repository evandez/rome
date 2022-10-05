"""Make knowns datasets to evaluate context mediation."""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import requests
from tqdm.auto import tqdm

KnownsDatasets = dict[tuple[str, ...], list[dict]]

KEY_MEDIATED = "med"
KEY_UNMEDIATED = "umed"
KEY_SUBJECT = "subj"
KEY_ATTRIBUTE = "attr"

WINOVENTI_URL = "https://raw.githubusercontent.com/commonsense-exception/commonsense-exception/main/data/winoventi_bert_large_final.tsv"


def uncapitalize(string: str) -> str:
    """Uncapitalize the string."""
    return string[0].lower() + string[1:]


def make_counterfact(data_dir: Path) -> KnownsDatasets:
    """Convert CounterFact into knowns datasets for mediation."""
    import dsets

    counterfact = dsets.CounterFactDataset(str(data_dir))
    datasets: KnownsDatasets = defaultdict(list)
    known_id = 0
    for cf_sample in tqdm(counterfact, desc="make counterfact"):
        cf_requested_rewrite = cf_sample["requested_rewrite"]
        cf_subject = cf_requested_rewrite["subject"]
        cf_target_new = cf_requested_rewrite["target_new"]
        cf_target_true = cf_requested_rewrite["target_true"]
        cf_prompt = cf_requested_rewrite["prompt"]
        cf_generation_prompt = cf_sample["generation_prompts"][0]
        for target_mediation in (KEY_MEDIATED, KEY_UNMEDIATED):
            for target_entity, target_occurrence in (
                (KEY_SUBJECT, 0),
                (KEY_SUBJECT, 1),
                (KEY_ATTRIBUTE, 0),
            ):
                key = (target_mediation, target_entity, str(target_occurrence))
                sample = {"known_id": known_id}
                known_id += 1

                if target_entity == KEY_SUBJECT:
                    subject = cf_subject
                else:
                    assert target_entity == KEY_ATTRIBUTE
                    subject = cf_target_new
                sample["subject"] = subject

                sample["occurrence"] = target_occurrence

                if target_mediation == KEY_MEDIATED:
                    attribute = cf_target_new
                else:
                    assert target_mediation == KEY_UNMEDIATED
                    attribute = cf_target_true
                sample["attribute"] = attribute

                prompt = cf_generation_prompt
                if not prompt.startswith(cf_subject):
                    prompt = uncapitalize(prompt)
                prompt = f"Suppose {prompt} {cf_target_new}. {cf_prompt}"
                sample["prompt"] = prompt

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
    for wv_sample in tqdm(wv_samples):
        wv_masked_prompt = wv_sample["masked_prompt"]
        wv_biased_word_context = wv_sample["biased_word_context"]
        wv_adv_word_context = wv_sample["adversarial_word_context"]
        wv_word = wv_sample["Word"]
        wv_target = wv_sample["target"]
        wv_incorrect = wv_sample["incorrect"]
        for prompt, attribute, target in (
            (wv_masked_prompt, wv_adv_word_context, wv_target),
            (
                wv_masked_prompt.replace(wv_adv_word_context, wv_biased_word_context),
                wv_biased_word_context,
                wv_incorrect,
            ),
        ):
            sample = {
                "known_id": known_id,
                "prompt": prompt.replace("[MASK]", "").rstrip(". "),
                "attribute": target,
            }
            known_id += 1

            for key, subject, occurrence in (
                (KEY_SUBJECT, wv_word, 0),
                (KEY_SUBJECT, wv_word, 1),
                (KEY_ATTRIBUTE, attribute, 0),
            ):
                datasets[key, str(occurrence)].append(
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
        for keys, knowns_dataset in knowns_datasets.items():
            knowns_dataset = random.sample(knowns_dataset, k=args.sample)
            knowns_dataset_file = output_dir / f"{dataset}_{'_'.join(keys)}.json"
            with knowns_dataset_file.open("w") as handle:
                json.dump(knowns_dataset, handle)


if __name__ == "__main__":
    main()
