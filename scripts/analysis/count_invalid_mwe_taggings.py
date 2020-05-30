import argparse
import json
import random
import re

from tqdm import tqdm


RE_TAGGING = re.compile(r'^(O|B(o|b(i[_~])+|I[_~])*(I[_~])+)+$')


def main(predictions_path):
    print(f"Reading AllenNLP predictions from: {predictions_path}")
    num_total = 0
    num_valid = 0

    invalid_taggings = []
    with open(predictions_path, "r") as predictions_file:
        for line in tqdm(predictions_file):
            tags = json.loads(line)["tags"]
            tokens = json.loads(line)["tokens"]
            mwetags = [tag.split("-")[0] for tag in tags]
            if RE_TAGGING.match(''.join(mwetags)):
                num_valid += 1
            else:
                invalid_taggings.append((tokens, tags))
            num_total += 1
    num_invalid = num_total - num_valid
    print(f"Number of total taggings: {num_total}")
    print(f"Number of valid taggings: {num_valid}")
    print(f"Proportion of valid taggings: {num_valid / num_total}")
    print(f"Number of invalid taggings: {num_invalid}")
    print(f"Proportion of invalid taggings: {num_invalid / num_total}")
    random.shuffle(invalid_taggings)
    print("5 random invalid taggings:")
    for tokens, tags in invalid_taggings[:5]:
        print(f"Tokens: {tokens}")
        print(f"Tags: {tags}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Given an AllenNLP prediction jsonl file, "
                     "count the number of invalid MWE taggings."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--predictions-path", required=True,
                        help="Path to the predictions file.")
    args = parser.parse_args()
    main(args.predictions_path)
