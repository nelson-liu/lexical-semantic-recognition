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


def is_int(x):
    try:
        int(x)
        return True
    except ValueError:
        return False


def main(data_path, output_path):
    print(f"Reading STREUSLE conllulex data from: {data_path}")
    gold_lextags = []

    with open(data_path) as data_file:
        for is_divider, lines in itertools.groupby(data_file, _is_divider):
            # Ignore the divider chunks, so that `lines` corresponds to the words
            # of a single sentence.
            if not is_divider:
                instance_gold_lextags = []
                stripped_split_lines = [(line.strip().split() if not line.startswith("#") else
                                         line.strip()) for line in lines]
                # Remove lines where the first index cannot be coerced to an int, since sometimes there are
                # "copy" tokens that have indices like 7.1
                stripped_split_lines = [line for line in stripped_split_lines if is_int(line[0])]
                for stripped_split_line in stripped_split_lines:
                    if stripped_split_line[0] != "#":
                        instance_gold_lextags.append(stripped_split_line[-1])
                gold_lextags.append(instance_gold_lextags)

    output_lines = []
    prediction_index = 0
    output_lines.append("# global.columns = ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC PARSEME:MWE\n")
    # Write the predictions to the original file.
    with open(data_path) as data_file:
        for is_divider, lines in itertools.groupby(data_file, _is_divider):
            # Ignore the divider chunks, so that `lines` corresponds to the words
            # of a single sentence.
            if not is_divider:
                instance_gold_lextags = gold_lextags[prediction_index]
                stripped_split_lines = [(line.strip().split() if not line.startswith("#") else
                                         line.strip()) for line in lines]
                # Remove lines where the first index cannot be coerced to an int, since sometimes there are
                # "copy" tokens that have indices like 7.1
                stripped_split_lines = [line for line in stripped_split_lines if is_int(line[0])]
                # Get the sequence of predicted VMWE tags from the
                # instance predictions
                assert len(instance_gold_lextags) == len([x[0] for x in stripped_split_lines if not x[0].startswith("#")])
                vmwe_gold = get_vmwe_predictions_from_lextags(instance_gold_lextags)

                tag_index = 0
                for stripped_split_line in stripped_split_lines:
                    if stripped_split_line[0] == "#":
                        # Output the line as is
                        output_lines.append(stripped_split_line + "\n")
                    else:
                        # Take the first time columns wholesale
                        output_split_line = stripped_split_line[:10]
                        # Append on the VMWE label
                        output_split_line.append(vmwe_gold[tag_index])
                        output_lines.append("\t".join(output_split_line) + "\n")
                        tag_index += 1
                output_lines.append("\n")
                prediction_index += 1

    print(f"Writing output to {output_path}")
    with open(output_path, "w") as output_file:
        for line in output_lines:
            output_file.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Convert a gold STREUSLE CONLLULEX format "
                     "suitable for the PARSEME evaluator."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-path", required=True,
                        help="Path to the data file to convert.")
    parser.add_argument("--output-path", required=True,
                        help=("Path to write file in the PARSEME format."))
    args = parser.parse_args()
    main(args.data_path, args.output_path)
