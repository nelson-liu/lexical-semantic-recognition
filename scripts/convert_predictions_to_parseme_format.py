from typing import List
import argparse
import itertools
import json

from tqdm import tqdm


def _is_divider(line: str) -> bool:
    return line.strip() == ""


def get_vmwe_predictions_from_lextags(lextags: List[str]) -> List[List[str]]:
    vmwe_predictions = []
    current_vmwe_index = 1
    for lextag_idx, lextag in enumerate(lextags):
        split_lextag = lextag.split("-")
        mwe_tag = split_lextag[0]
        if mwe_tag == "O" or mwe_tag == "o":
            # Current token is not a MWE, so we just predict *
            vmwe_predictions.append("*")
            continue
        if (len(split_lextag) > 1 and
                not split_lextag[1].startswith("V.") and
                not split_lextag[1] == "V"):
            # This is not a verbal MWE, predict *
            vmwe_predictions.append("*")
            continue
        else:
            # VMWE here, extract it and prepend identifier.
            if mwe_tag == "B":
                vmwe_subtype = split_lextag[1][2:]
                # Starting a new MWE, but we don't know if it's weak or strong.
                # So we look ahead. If we see I~, it's weak. If we see I_, it's strong
                is_strong_mwe = None
                for lookahead_lextag in lextags[lextag_idx+1:]:
                    if lookahead_lextag.split("-")[0] == "I_":
                        is_strong_mwe = True
                        break
                    elif lookahead_lextag.split("-")[0] == "I~":
                        is_strong_mwe = False
                        break
                    elif lookahead_lextag.split("-")[0] == "B":
                        raise ValueError(f"Encountered B when we we looking for I_ or I~. Lextags {lextags}")
                if is_strong_mwe is None:
                    raise ValueError(f"Didn't find I_ or I~ we were looking for. index {lextag_idx}, Lextags {lextags}")
                if is_strong_mwe:
                    vmwe_predictions.append(f"{current_vmwe_index}:{vmwe_subtype}")
                    current_vmwe_index += 1
                else:
                    vmwe_predictions.append("*")
            elif mwe_tag == "I_":
                # Continuing the most recent (strong) MWE (B)
                # Get the index of the last vmwe
                previous_vmwe_index = -1
                for idx, vmwe_prediction in reversed(list(enumerate(vmwe_predictions))):
                    if vmwe_prediction != "*" and lextags[idx].split("-")[0] == "B":
                        previous_vmwe_index = int(vmwe_prediction.split(":")[0])
                        break
                    if (lextags[idx].split("-")[0] == "B" and
                            len(lextags[idx].split("-")) > 1 and
                            not lextags[idx].split("-")[1].startswith("V.") and
                            not lextags[idx].split("-")[1] == "V"):
                        # We found a B- tag that isn't a strong VMWE
                        break
                if previous_vmwe_index != -1:
                    vmwe_predictions.append(f"{previous_vmwe_index}")
                else:
                    vmwe_predictions.append("*")
            elif mwe_tag == "I~":
                # Continuing the most recent (weak) MWE (B).
                # Since we looked ahead, the B should have been assigned *,
                # so we can similarly use * here.
                vmwe_predictions.append("*")
            elif mwe_tag == "b":
                vmwe_subtype = split_lextag[1][2:]
                # Starting a new inner MWE, but we don't know if it's weak or strong.
                # So we look ahead. If we see i~, it's weak. If we see i_, it's strong
                is_strong_mwe = None
                for lookahead_lextag in lextags[lextag_idx+1:]:
                    if lookahead_lextag.split("-")[0] == "i_":
                        is_strong_mwe = True
                        break
                    elif lookahead_lextag.split("-")[0] == "i~":
                        is_strong_mwe = False
                        break
                    elif lookahead_lextag.split("-")[0] == "b":
                        raise ValueError(f"Encountered b when we we looking for i_ or i~. Lextags {lextags}")
                    elif lookahead_lextag.split("-")[0] == "B":
                        raise ValueError(f"Encountered B when we we looking for i_ or i~. Lextags {lextags}")
                if is_strong_mwe is None:
                    raise ValueError(f"Didn't find i_ or i~ we were looking for. index {lextag_idx}, Lextags {lextags}")
                if is_strong_mwe:
                    vmwe_predictions.append(f"{current_vmwe_index}:{vmwe_subtype}")
                    current_vmwe_index += 1
                else:
                    vmwe_predictions.append("*")
            elif mwe_tag == "i_":
                # Continuing the most recent inner strong MWE (b)
                # Get the index of the last b vmwe
                previous_vmwe_index = -1
                for idx, vmwe_prediction in reversed(list(enumerate(vmwe_predictions))):
                    if vmwe_prediction != "*" and lextags[idx].split("-")[0] == "b":
                        previous_vmwe_index = int(vmwe_prediction.split(":")[0])
                        break
                    if (lextags[idx].split("-")[0] == "B" and
                            len(lextags[idx].split("-")) > 1 and
                            not lextags[idx].split("-")[1].startswith("V.") and
                            not lextags[idx].split("-")[1] == "V"):
                        # We found a B- tag that isn't a strong VMWE
                        break
                if previous_vmwe_index != -1:
                    vmwe_predictions.append(f"{previous_vmwe_index}")
                else:
                    vmwe_predictions.append("*")
            elif mwe_tag == "i~":
                # Continuing the most recent (weak) inner MWE (b).
                # Since we looked ahead, the b should have been assigned *,
                # so we can similarly use * here.
                vmwe_predictions.append("*")
    assert len(vmwe_predictions) == len(lextags)
    return vmwe_predictions


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
                stripped_split_lines = [(line.strip().split() if not line.startswith("#") else
                                         line.strip()) for line in lines]
                # Get the sequence of predicted VMWE tags from the
                # instance predictions
                assert len(instance_predictions) == len([x[0] for x in stripped_split_lines if not x[0].startswith("#")])
                vmwe_predictions = get_vmwe_predictions_from_lextags(instance_predictions)
                tag_index = 0
                for stripped_split_line in stripped_split_lines:
                    if stripped_split_line[0] == "#":
                        # Output the line as is
                        output_lines.append(stripped_split_line + "\n")
                    else:
                        stripped_split_line[-1] = vmwe_predictions[tag_index]
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
                     "suitable for the PARSEME evaluator."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--predictions-path", required=True,
                        help="Path to the predictions file.")
    parser.add_argument("--test-data-path", required=True,
                        help=("Path to test data file from which the "
                              "predictions were generated."))
    parser.add_argument("--output-path", required=True,
                        help=("Path to write file in the PARSEME format."))
    args = parser.parse_args()
    main(args.predictions_path, args.test_data_path, args.output_path)
