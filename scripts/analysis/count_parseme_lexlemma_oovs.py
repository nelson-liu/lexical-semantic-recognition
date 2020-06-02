import argparse
from collections import defaultdict, Counter
import itertools
from conllu import parse_incr


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
    train_lexlemmas = []
    with open(train_path) as data_file:
        for is_divider, lines in itertools.groupby(data_file, _is_divider):
            # Ignore the divider chunks, so that `lines` corresponds to the words
            # of a single sentence.
            if not is_divider:
                instance_lexlemmas = []
                stripped_split_lines = [(line.strip().split("\t") if not line.startswith("#") else
                                         line.strip()) for line in lines]
                # Remove lines where the first index cannot be coerced to an int, since sometimes there are
                # "copy" tokens that have indices like 7.1
                stripped_split_lines = [line for line in stripped_split_lines if is_int(line[0])]
                for stripped_split_line in stripped_split_lines:
                    if stripped_split_line[0] != "#":
                        instance_lexlemmas.append(stripped_split_line[12] if
                                                  stripped_split_line[12] != "_" else
                                                  stripped_split_line[2])
                train_lexlemmas.append(instance_lexlemmas)
                train_vocab.update(instance_lexlemmas)

    print(f"Reading test data from {test_path}")
    parseme_jsonl = []
    with open(test_path, "r") as conllu_file:
        for annotation in parse_incr(conllu_file):
            tokens = [x["lemma"] for x in annotation]
            mwes = [x["parseme:mwe"] for x in annotation]
            parseme_jsonl.append({
                "lemmas": tokens,
                "mwes": mwes
            })

    test_lexlemmas = []
    test_lexcats = []
    for instance in parseme_jsonl:
        instance_lexlemmas = []
        instance_lexcats = []
        test_example_lemmas = instance["lemmas"]
        test_example_mwes = instance["mwes"]
        instance_mwe_types = {}
        index_to_mwes = defaultdict(list)
        for seq_index, (test_example_lemma, test_example_mwe) in enumerate(zip(test_example_lemmas, test_example_mwes)):
            if test_example_mwe == "*":
                categories = ["*"]
            elif ";" in test_example_mwe:
                categories = []
                for mwe in test_example_mwe.split(";"):
                    if len(mwe) == 1:
                        category = instance_mwe_types[int(mwe)]
                        index_to_mwes[int(mwe)].append((seq_index, test_example_lemma))
                    else:
                        index = mwe.split(":")[0]
                        category = mwe.split(":")[1]
                        instance_mwe_types[int(index)] = category
                        index_to_mwes[int(index)].append((seq_index, test_example_lemma))
                    categories.append(category)
            elif len(test_example_mwe) > 1:
                categories = []
                index = test_example_mwe.split(":")[0]
                category = test_example_mwe.split(":")[1]
                instance_mwe_types[int(index)] = category
                index_to_mwes[int(index)].append((seq_index, test_example_lemma))
                categories.append(category)
            else:
                categories = []
                category = instance_mwe_types[int(test_example_mwe)]
                index_to_mwes[int(test_example_mwe)].append((seq_index, test_example_lemma))
                categories.append(category)
            instance_lexcats.append(categories)
            instance_lexlemmas.append([test_example_lemma])

        # Edit the instance lexlemmas to replace MWE constituent tokens with the full MWE
        indices_to_replace = set()
        for mwe in index_to_mwes.values():
            mwe_indices = [x[0] for x in mwe]
            indices_to_replace.update(mwe_indices)
        for index in indices_to_replace:
            # Get the MWEs that have this index:
            index_mwes = []
            for mwe in index_to_mwes.values():
                mwe_indices = [x[0] for x in mwe]
                if index in mwe_indices:
                    mwe_string = " ".join([x[1] for x in mwe])
                    index_mwes.append(mwe_string)
            instance_lexlemmas[index] = index_mwes
        test_lexlemmas.append(instance_lexlemmas)
        test_lexcats.append(instance_lexcats)

    # Count the number of OOV tokens
    lexcat_oovs = defaultdict(Counter)
    num_test_tokens = 0
    num_oov_test_tokens = 0
    for test_example_lexlemma, test_example_lexcats in zip(test_lexlemmas, test_lexcats):
        for test_lexlemma, test_lexcat in zip(test_example_lexlemma, test_example_lexcats):
            for lexlemma in test_lexlemma:
                if lexlemma not in train_vocab:
                    num_oov_test_tokens += 1
                    for lexcat in test_lexcat:
                        lexcat_oovs[lexcat]["oov"] += 1
                num_test_tokens += 1
                for lexcat in test_lexcat:
                    lexcat_oovs[lexcat]["total"] += 1
    print(f"Number of test tokens: {num_test_tokens}")
    print(f"OOV test tokens: {num_oov_test_tokens / num_test_tokens} "
          f"({num_oov_test_tokens} / {num_test_tokens})")
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
                        help="Path to the test PARSEME data file.")
    args = parser.parse_args()
    main(args.train_path, args.test_path)
