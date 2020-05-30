import argparse
from collections import Counter
import json
import re

from tqdm import tqdm


RE_TAGGING = re.compile(r'^(O|B(o|b(i[_~])+|I[_~])*(I[_~])+)+$')


def main(gold_path):
    print(f"Reading gold data from {gold_path}")
    with open(gold_path, "r") as gold_file:
        tagging_data = json.load(gold_file)
    gold_tags = []
    for instance in tqdm(tagging_data):
        labels = [x["lextag"] for x in instance["toks"]]
        gold_tags.append(labels)

    per_lexcat_total_counts = Counter()
    per_lexcat_mwe_counts = Counter()
    for gold in gold_tags:
        for gold_word in gold:
            # Split off the lexcat
            split_gold = gold_word.split("-")
            if len(split_gold) > 1:
                lexcat = split_gold[1]
            else:
                # No lexcat
                continue
            if split_gold[0].lower() != "o":
                per_lexcat_mwe_counts[lexcat] += 1
            per_lexcat_total_counts[lexcat] += 1
    for lexcat in per_lexcat_total_counts:
        print(f"{lexcat}: {per_lexcat_mwe_counts[lexcat] / per_lexcat_total_counts[lexcat]} ({per_lexcat_mwe_counts[lexcat]} / {per_lexcat_total_counts[lexcat]})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Given an AllenNLP prediction jsonl file and a gold data file, "
                     "calculate the token-level tag exact match."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gold-path", required=True,
                        help="Path to the gold data file.")
    args = parser.parse_args()
    main(args.gold_path)
