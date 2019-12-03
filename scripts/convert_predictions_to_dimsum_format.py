from typing import List
import argparse
import itertools
import json

from tqdm import tqdm

from dimsum_mwe_simplify import simplify

def _is_divider(line: str) -> bool:
    return line.strip() == ""

def get_dimsum_predictions_from_lextags(lextags: List[str]) -> List[List[str]]:
    simplified_mwe_tags = simplify([x.split("-")[0] for x in lextags],
                                   simplification="weak",
                                   policy="all")[0]
    supersense_tags = []
    for lextag in lextags:
        split_lextag = lextag.split("-")
        if len(split_lextag) > 2:
            supersense = split_lextag[2]
            if supersense.startswith("v.") or supersense.startswith("n"):
                supersense_tags.append(supersense.lower())
            else:
                supersense_tags.append("")
        else:
            supersense_tags.append("")
    # Get the MWE offsets from the simplified MWE tags:
    mwe_offsets = []
    for mwe_tag_index, mwe_tag in enumerate(simplified_mwe_tags):
        if mwe_tag == "O" or mwe_tag == "o":
            mwe_offsets.append(0)
        elif mwe_tag == "b" or mwe_tag == "B":
            # dimsum takes 1-indexed.
            mwe_offsets.append(0)
        elif mwe_tag == "i":
            # Go find the previous "b" or "i"
            found_mwe_continuation = False
            for i in reversed(range(0, mwe_tag_index)):
                if simplified_mwe_tags[i] in {"b", "i"}:
                    mwe_offsets.append(i + 1)
                    found_mwe_continuation = True
                    break
            if not found_mwe_continuation:
                raise ValueError(f"Didn't find MWE continuation (b or i) for "
                                 f"index {mwe_tag_index} tag {mwe_tag}. "
                                 f"MWE tags: {simplified_mwe_tags}")
        elif mwe_tag == "I":
            # Go find the previous "b" or "i"
            found_mwe_continuation = False
            for i in reversed(range(0, mwe_tag_index)):
                if simplified_mwe_tags[i] in {"B", "I"}:
                    mwe_offsets.append(i + 1)
                    found_mwe_continuation = True
                    break
            if not found_mwe_continuation:
                raise ValueError(f"Didn't find MWE continuation (B or I) for "
                                 f"index {mwe_tag_index} tag {mwe_tag}. "
                                 f"MWE tags: {simplified_mwe_tags}")
    assert len(simplified_mwe_tags) == len(mwe_offsets)
    assert len(mwe_offsets) == len(supersense_tags)
    return simplified_mwe_tags, mwe_offsets, supersense_tags

def main(predictions_path, test_data_path, output_path):
    print(f"Reading AllenNLP predictions from: {predictions_path}")
    predictions = []

    with open(predictions_path, "r") as predictions_file:
        for line in tqdm(predictions_file):
            predictions.append(json.loads(line)["tags"])

    output_lines = []
    prediction_index = 0
    # Write the predictions to the original file.
    with open(test_data_path) as test_data_file:
        for is_divider, lines in itertools.groupby(test_data_file, _is_divider):
            # Ignore the divider chunks, so that `lines` corresponds to the words
            # of a single sentence.
            if not is_divider:
                instance_predictions = predictions[prediction_index]
                stripped_split_lines = [(line.strip().split("\t") if not line.startswith("#") else
                                         line.strip()) for line in lines]
                # Get the sequence of predicted VMWE tags from the
                # instance predictions
                assert len(instance_predictions) == len([x[0] for x in stripped_split_lines if not x[0].startswith("#")])
                pred_mwe_tags, mwe_offsets, pred_supersense_tags = get_dimsum_predictions_from_lextags(instance_predictions)
                tag_index = 0
                for stripped_split_line in stripped_split_lines:
                    if stripped_split_line[0] == "#":
                        # Output the line as is
                        output_lines.append(stripped_split_line + "\n")
                    else:
                        stripped_split_line[4] = pred_mwe_tags[tag_index]
                        stripped_split_line[5] = str(mwe_offsets[tag_index])
                        stripped_split_line[7] = pred_supersense_tags[tag_index]
                        output_lines.append("\t".join(stripped_split_line) + "\n")
                        tag_index += 1
                output_lines.append("\n")
                prediction_index += 1

    print(f"Writing output to {output_path}")
    with open(output_path, "w") as output_file:
        for line in output_lines:
            output_file.write(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Convert an AllenNLP prediction jsonl file into a format "
                     "suitable for the DIMSUM evaluator."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--predictions-path", required=True,
                        help="Path to the predictions file.")
    parser.add_argument("--test-data-path", required=True,
                        help=("Path to test data file from which the "
                              "predictions were generated."))
    parser.add_argument("--output-path", required=True,
                        help=("Path to write file in the DIMSUM format."))
    args = parser.parse_args()
    main(args.predictions_path, args.test_data_path, args.output_path)
