from typing import List
import argparse
import itertools
import json

from tqdm import tqdm

def _is_divider(line: str) -> bool:
    return line.strip() == ""

def get_streusle2_predictions_from_tags(tags: List[str]) -> List[List[str]]:
    # Get the MWE offsets:
    mwe_offsets = []
    mwe_tags = [tag.split("-")[0] for tag in tags]
    for mwe_tag_index, mwe_tag in enumerate(mwe_tags):
        if mwe_tag in ("O", "o"):
            mwe_offsets.append(0)
        elif mwe_tag in ("B", "b"):
            mwe_offsets.append(0)
        elif mwe_tag in ("ĩ", "ī"):
            # Go find the previous "b" or "ĩ" or "ī"
            found_mwe_continuation = False
            for i in reversed(range(0, mwe_tag_index)):
                if mwe_tags[i] in {"b", "ĩ", "ī"}:
                    mwe_offsets.append(i + 1)
                    found_mwe_continuation = True
                    break
            if not found_mwe_continuation:
                raise ValueError(f"Didn't find MWE continuation (b or i) for "
                                 f"index {mwe_tag_index} tag {mwe_tag}. "
                                 f"MWE tags: {mwe_tags}")
        elif mwe_tag in ("Ĩ", "Ī"):
            # Go find the previous "B" or "Ĩ" or "Ī"
            found_mwe_continuation = False
            for i in reversed(range(0, mwe_tag_index)):
                if mwe_tags[i] in {"B", "Ĩ", "Ī"}:
                    mwe_offsets.append(i + 1)
                    found_mwe_continuation = True
                    break
            if not found_mwe_continuation:
                raise ValueError(f"Didn't find MWE continuation (B or I) for "
                                 f"index {mwe_tag_index} tag {mwe_tag}. "
                                 f"MWE tags: {mwe_tags}")
    # Get the MWE strength
    mwe_strengths = []
    for mwe_tag in mwe_tags:
        if mwe_tag in ("O", "o"):
            mwe_strengths.append("")
        elif mwe_tag in ("B", "b"):
            mwe_strengths.append("")
        elif mwe_tag in ("ĩ", "Ĩ"):
            mwe_strengths.append("~")
        elif mwe_tag in ("ī", "Ī"):
            mwe_strengths.append("_")

    # Get the class tags
    class_tags = []
    for tag in tags:
        split_tag = tag.split("-")
        if len(split_tag) > 2:
            class_tag = split_tag[1]
            class_tags.append(class_tag)
        else:
            class_tags.append("")
    assert len(mwe_tags) == len(mwe_offsets)
    assert len(mwe_offsets) == len(class_tags)
    assert len(mwe_strengths) == len(class_tags)
    return mwe_offsets, mwe_strengths, class_tags

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
                mwe_offsets, mwe_strengths, class_tags = get_streusle2_predictions_from_tags(
                    instance_predictions)
                tag_index = 0
                for stripped_split_line in stripped_split_lines:
                    if stripped_split_line[0] == "#":
                        # Output the line as is
                        output_lines.append(stripped_split_line + "\n")
                    else:
                        # Insert the predictions into the line
                        stripped_split_line[4] = instance_predictions[tag_index]
                        stripped_split_line[5] = str(mwe_offsets[tag_index])
                        stripped_split_line[6] = str(mwe_strengths[tag_index])
                        stripped_split_line[7] = str(class_tags[tag_index])
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
                         "suitable for the STREUSLE2.0 evaluator."),
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--predictions-path", required=True,
                        help="Path to the predictions file.")
    parser.add_argument("--test-data-path", required=True,
                        help=("Path to test data file (.tags) from which the "
                              "predictions were generated."))
    parser.add_argument("--output-path", required=True,
                        help=("Path to write file in the STREUSLE2.0 .tags format."))
    args = parser.parse_args()
    main(args.predictions_path, args.test_data_path, args.output_path)
