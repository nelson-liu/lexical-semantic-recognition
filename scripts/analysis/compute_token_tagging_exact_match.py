import argparse
import json
import re

from tqdm import tqdm


RE_TAGGING = re.compile(r'^(O|B(o|b(i[_~])+|I[_~])*(I[_~])+)+$')


def main(predictions_path, gold_path):
    print(f"Reading AllenNLP predictions from: {predictions_path}")

    predictions = []
    with open(predictions_path, "r") as predictions_file:
        for line in tqdm(predictions_file):
            tags = json.loads(line)["tags"]
            predictions.append(tags)

    print(f"Reading gold data from {gold_path}")
    with open(gold_path, "r") as gold_file:
        tagging_data = json.load(gold_file)
    gold_tags = []
    for instance in tqdm(tagging_data):
        labels = [x["lextag"] for x in instance["toks"]]
        gold_tags.append(labels)

    # Calculate exact match.
    assert len(predictions) == len(gold_tags)
    num_total = 0
    num_exact_match = 0
    num_exact_match_minus_lexcat = 0
    num_exact_match_minus_supersense = 0
    for prediction, gold in zip(predictions, gold_tags):
        for predicted_word, gold_word in zip(prediction, gold):
            if predicted_word == gold_word:
                num_exact_match += 1
            # Split off the lexcat
            predicted_minus_lexcat = predicted_word.split("-")
            if len(predicted_minus_lexcat) > 1:
                predicted_minus_lexcat.pop(1)
            gold_minus_lexcat = gold_word.split("-")
            if len(gold_minus_lexcat) > 1:
                gold_minus_lexcat.pop(1)
            if predicted_minus_lexcat == gold_minus_lexcat:
                num_exact_match_minus_lexcat += 1
            # Split off the supersense
            predicted_minus_supersense = predicted_word.split("-")
            if len(predicted_minus_supersense) > 2:
                predicted_minus_supersense.pop(2)
            gold_minus_supersense = gold_word.split("-")
            if len(gold_minus_supersense) > 2:
                gold_minus_supersense.pop(2)
            if predicted_minus_supersense == gold_minus_supersense:
                num_exact_match_minus_supersense += 1
            num_total += 1

    print(f"Num total: {num_total}")
    print(f"Exact Match: {num_exact_match / num_total}")
    print(f"Exact Match (minus lexcat): {num_exact_match_minus_lexcat / num_total}")
    print(f"Exact Match (minus supersense): {num_exact_match_minus_supersense / num_total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Given an AllenNLP prediction jsonl file and a gold data file, "
                     "calculate the token-level tag exact match."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--predictions-path", required=True,
                        help="Path to the predictions file.")
    parser.add_argument("--gold-path", required=True,
                        help="Path to the gold data file.")
    args = parser.parse_args()
    main(args.predictions_path, args.gold_path)
