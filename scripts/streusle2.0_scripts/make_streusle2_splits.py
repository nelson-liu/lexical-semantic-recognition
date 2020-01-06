import argparse
import random

from tqdm import tqdm

random.seed(0)

def main(data_path,
         train_ids_path,
         test_ids_path,
         num_dev_sentences,
         train_output_path,
         test_output_path,
         dev_output_path):
    print(f"Reading train IDs from: {train_ids_path}")
    train_ids = set()
    with open(train_ids_path) as train_ids_file:
        for line in tqdm(train_ids_file):
            train_ids.add(line.strip("\n"))

    if num_dev_sentences >= len(train_ids):
        raise ValueError(f"Specified dev set of {num_dev_sentences} "
                         "sentences, but only {len(train_ids)} "
                         "training instances")

    print(f"Reading test IDs from: {test_ids_path}")
    test_ids = set()
    with open(test_ids_path) as test_ids_file:
        for line in tqdm(test_ids_file):
            test_ids.add(line.strip("\n"))

    print(f"Reading STREUSLE 2.0 data from: {data_path}")
    train_data = []
    test_data = []

    with open(data_path, "r") as data_file:
        for line in tqdm(data_file):
            line = line.strip("\n")
            example_id = line.split("\t")[0]
            if example_id in train_ids:
                train_data.append(line)
            elif example_id in test_ids:
                test_data.append(line)
            else:
                raise ValueError(f"example id {example_id} not in train or test ids")

    # Shuffle the train data, and subsample for development.
    random.shuffle(train_data)
    dev_data = train_data[:num_dev_sentences]
    train_data = train_data[num_dev_sentences:]

    # Write the split data to the output files
    print(f"Writing train split to {train_output_path}")
    with open(train_output_path, "w") as train_output_file:
        for line in train_data:
            train_output_file.write(f"{line}\n")
    print(f"Writing dev split to {dev_output_path}")
    with open(dev_output_path, "w") as dev_output_file:
        for line in dev_data:
            dev_output_file.write(f"{line}\n")
    print(f"Writing test split to {test_output_path}")
    with open(test_output_path, "w") as test_output_file:
        for line in test_data:
            test_output_file.write(f"{line}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Split the STREUSLE 2.0 dataset into training "
                     "and test splits."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-path", required=True, type=str,
                        help="Path to the streusle.sst file.")
    parser.add_argument("--train-ids-path", required=True, type=str,
                        help=("Path to ids for use in the training split."))
    parser.add_argument("--test-ids-path", required=True, type=str,
                        help=("Path to ids for use in the test split."))
    parser.add_argument("--num-dev-sentences", required=True, type=int,
                        help=("Number of sentences to subsample from train "
                              "to use as a development set."))
    parser.add_argument("--train-output-path", required=True, type=str,
                        help=("Path to write training dataset."))
    parser.add_argument("--test-output-path", required=True, type=str,
                        help=("Path to write test dataset."))
    parser.add_argument("--dev-output-path", required=True, type=str,
                        help=("Path to write dev dataset."))
    args = parser.parse_args()
    main(args.data_path,
         args.train_ids_path,
         args.test_ids_path,
         args.num_dev_sentences,
         args.train_output_path,
         args.test_output_path,
         args.dev_output_path)
