import argparse
from collections import defaultdict, Counter
import itertools
import json

from tqdm import tqdm


def is_int(x):
    try:
        int(x)
        return True
    except ValueError:
        return False


def _is_divider(line: str) -> bool:
    return line.strip() == ""


def main(train_path, test_path):
    train_vocab = set()
    print(f"Reading train data from {train_path}")
    train_tags = []
    train_lexlemmas = []
    with open(train_path) as data_file:
        for is_divider, lines in itertools.groupby(data_file, _is_divider):
            # Ignore the divider chunks, so that `lines` corresponds to the words
            # of a single sentence.
            if not is_divider:
                instance_lexlemmas = []
                instance_lextags = []
                stripped_split_lines = [(line.strip().split() if not line.startswith("#") else
                                         line.strip()) for line in lines]
                # Remove lines where the first index cannot be coerced to an int, since sometimes there are
                # "copy" tokens that have indices like 7.1
                stripped_split_lines = [line for line in stripped_split_lines if is_int(line[0])]
                for stripped_split_line in stripped_split_lines:
                    if stripped_split_line[0] != "#":
                        instance_lextags.append(stripped_split_line[-1])
                        instance_lexlemmas.append(stripped_split_line[12])
                train_tags.append(instance_lextags)
                train_lexlemmas.append(instance_lexlemmas)
                train_vocab.update(instance_lexlemmas)

    print(f"Reading test data from {test_path}")
    test_tags = []
    test_lexlemmas = []
    test_lexcats = []
    with open(test_path) as data_file:
        for is_divider, lines in itertools.groupby(data_file, _is_divider):
            # Ignore the divider chunks, so that `lines` corresponds to the words
            # of a single sentence.
            if not is_divider:
                instance_lexlemmas = []
                instance_lextags = []
                instance_lexcats = []
                stripped_split_lines = [(line.strip().split() if not line.startswith("#") else
                                         line.strip()) for line in lines]
                # Remove lines where the first index cannot be coerced to an int, since sometimes there are
                # "copy" tokens that have indices like 7.1
                stripped_split_lines = [line for line in stripped_split_lines if is_int(line[0])]
                for stripped_split_line in stripped_split_lines:
                    if stripped_split_line[0] != "#":
                        instance_lextags.append(stripped_split_line[-1])
                        instance_lexlemmas.append(stripped_split_line[12])
                        instance_lexcats.append(stripped_split_line[11])
                test_tags.append(instance_lextags)
                test_lexlemmas.append(instance_lexlemmas)
                test_lexcats.append(instance_lexcats)

    # Count the number of OOV tokens
    lexcat_oovs = defaultdict(Counter)
    num_test_tokens = 0
    num_oov_test_tokens = 0
    num_oov_and_mwe_test_tokens = 0
    for test_example_words, test_example_lexcats, test_example_tags in zip(test_lexlemmas, test_lexcats, test_tags):
        for test_word, test_lexcat, test_tag in zip(test_example_words, test_example_lexcats, test_example_tags):
            if test_word not in train_vocab:
                num_oov_test_tokens += 1
                lexcat_oovs[test_lexcat]["oov"] += 1
                if test_tag.split("-")[0].lower() != "o":
                    num_oov_and_mwe_test_tokens += 1
            num_test_tokens += 1
            lexcat_oovs[test_lexcat]["total"] += 1
    print(f"Number of test tokens: {num_test_tokens}")
    print(f"OOV test tokens: {num_oov_test_tokens / num_test_tokens} "
          f"({num_oov_test_tokens} / {num_test_tokens})")
    print(f"OOV + MWE test tokens: {num_oov_and_mwe_test_tokens / num_test_tokens} "
          f"({num_oov_and_mwe_test_tokens} / {num_test_tokens})")
    print("OOV frequency for each lexcat:")
    for lexcat in lexcat_oovs:
        print(f"{lexcat}: {100 * lexcat_oovs[lexcat]['oov']/lexcat_oovs[lexcat]['total']:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Given a train file and a test file, count the number of OOV tokens, "
                     "the number of OOV tokens that are in MWEs."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-path", required=True,
                        help="Path to the train CONLLULEX data file.")
    parser.add_argument("--test-path", required=True,
                        help="Path to the test CONLLULEX data file.")
    args = parser.parse_args()
    main(args.train_path, args.test_path)
