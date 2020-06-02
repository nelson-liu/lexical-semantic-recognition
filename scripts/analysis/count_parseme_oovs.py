import argparse
from collections import defaultdict, Counter
import json

from conllu import parse_incr
from tqdm import tqdm


def main(train_path, test_path):
    train_vocab = set()
    print(f"Reading train data from {train_path}")
    with open(train_path, "r") as train_file:
        train_data = json.load(train_file)
    for instance in tqdm(train_data):
        tokens = [x["lexlemma"] for x in instance["toks"]]
        train_vocab.update(tokens)

    print(f"Reading test data from {test_path}")
    parseme_jsonl = []
    with open(test_path, "r") as conllu_file:
        for annotation in parse_incr(conllu_file):
            tokens = [x["form"] for x in annotation]
            mwes = [x["parseme:mwe"] for x in annotation]
            parseme_jsonl.append({
                "tokens": tokens,
                "mwes": mwes
            })

    # Count the number of OOV tokens
    category_oovs = defaultdict(Counter)
    num_test_tokens = 0
    num_oov_test_tokens = 0
    for test_instance in parseme_jsonl:
        test_example_words = test_instance["tokens"]
        test_example_mwes = test_instance["mwes"]
        instance_mwe_types = {}
        for test_word, test_mwe in zip(test_example_words, test_example_mwes):
            if test_mwe == "*":
                categories = ["*"]
            elif ";" in test_mwe:
                categories = []
                for mwe in test_mwe.split(";"):
                    if len(mwe) == 1:
                        category = instance_mwe_types[int(mwe)]
                    else:
                        index = mwe.split(":")[0]
                        category = mwe.split(":")[1]
                        instance_mwe_types[int(index)] = category
                    categories.append(category)
            elif len(test_mwe) > 1:
                categories = []
                index = test_mwe.split(":")[0]
                category = test_mwe.split(":")[1]
                instance_mwe_types[int(index)] = category
                categories.append(category)
            else:
                categories = []
                category = instance_mwe_types[int(test_mwe)]
                categories.append(category)
            if test_word not in train_vocab:
                num_oov_test_tokens += 1
                assert categories
                for category in categories:
                    category_oovs[category]["oov"] += 1
            num_test_tokens += 1
            assert categories
            for category in categories:
                category_oovs[category]["total"] += 1
    print(f"Number of test tokens: {num_test_tokens}")
    print(f"OOV test tokens: {num_oov_test_tokens / num_test_tokens} "
          f"({num_oov_test_tokens} / {num_test_tokens})")
    print("OOV frequency for each lexcat:")
    for category in category_oovs:
        print(f"{category}: {100 * category_oovs[category]['oov']/category_oovs[category]['total']:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Given a train file and a test file, count the number of OOV tokens, "
                     "the number of OOV tokens that are in MWEs."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--streusle-train-path", required=True,
                        help="Path to the STREUSLE train data file.")
    parser.add_argument("--parseme-test-path", required=True,
                        help="Path to the PARSEME test data file.")
    args = parser.parse_args()
    main(args.streusle_train_path, args.parseme_test_path)
