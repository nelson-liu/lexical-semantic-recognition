import argparse
from collections import Counter
import json
import re

from tqdm import tqdm


def main(train_path, test_path):
    train_vocab = set()
    print(f"Reading train data from {train_path}")
    with open(train_path, "r") as train_file:
        train_data = json.load(train_file)
    train_tags = []
    train_words = []
    for instance in tqdm(train_data):
        labels = [x["lextag"] for x in instance["toks"]]
        tokens = [x["word"] for x in instance["toks"]]
        train_tags.append(labels)
        train_words.append(tokens)
        train_vocab.update(tokens)

    print(f"Reading test data from {test_path}")
    with open(test_path, "r") as test_file:
        test_data = json.load(test_file)
    test_tags = []
    test_words = []
    for instance in tqdm(test_data):
        labels = [x["lextag"] for x in instance["toks"]]
        tokens = [x["word"] for x in instance["toks"]]
        test_tags.append(labels)
        test_words.append(tokens)

    # Count the number of OOV tokens
    num_test_tokens = 0
    num_oov_test_tokens = 0
    num_oov_and_mwe_test_tokens = 0
    for test_example_words, test_example_tags in zip(test_words, test_tags):
        for test_word, test_tag in zip(test_example_words, test_example_tags):
            if test_word not in train_vocab:
                num_oov_test_tokens += 1
                if test_tag.split("-")[0].lower() != "o":
                    num_oov_and_mwe_test_tokens += 1
            num_test_tokens += 1
    print(f"Number of test tokens: {num_test_tokens}")
    print(f"OOV test tokens: {num_oov_test_tokens / num_test_tokens} "
          f"({num_oov_test_tokens} / {num_test_tokens})")
    print(f"OOV + MWE test tokens: {num_oov_and_mwe_test_tokens / num_test_tokens} "
          f"({num_oov_and_mwe_test_tokens} / {num_test_tokens})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Given a train file and a test file, count the number of OOV tokens, "
                     "the number of OOV tokens that are in MWEs."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-path", required=True,
                        help="Path to the train data file.")
    parser.add_argument("--test-path", required=True,
                        help="Path to the test data file.")
    args = parser.parse_args()
    main(args.train_path, args.test_path)
