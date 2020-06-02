import argparse
from collections import defaultdict, Counter
import json
import re

from tqdm import tqdm


RE_TAGGING = re.compile(r'^(O|B(o|b(i[_~])+|I[_~])*(I[_~])+)+$')
# don't support plain I and i
STRENGTH = {'I_': '_', 'I~': '~', 'i_': '_', 'i~': '~', 'B': None, 'b': None, 'O': None, 'o': None}
# don't support plain I and i


def main(predictions_path, gold_path):
    print(f"Reading AllenNLP predictions from: {predictions_path}")

    predictions = []
    with open(predictions_path, "r") as predictions_file:
        for line in tqdm(predictions_file):
            tags = json.loads(line)["tags"]
            predictions.append(tags)

    print(f"Reading gold data from {gold_path}")
    with open(gold_path, "r") as gold_file:
        tagging_data = json.load(gold_file)
    gold_tags = []
    gold_tokens = []
    for instance in tqdm(tagging_data):
        labels = [x["lextag"] for x in instance["toks"]]
        tokens = [x["word"] for x in instance["toks"]]
        gold_tokens.append(tokens)
        gold_tags.append(labels)

    # Calculate P / R for each lexcat.
    assert len(predictions) == len(gold_tags)
    p_r_counter = defaultdict(Counter)
    for prediction, gold in zip(predictions, gold_tags):
        predicted_mwe_links = parse_mwe_links(prediction)
        gold_mwe_links = parse_mwe_links(gold)
        # Filter weak MWEs
        predicted_strong_mwe_links = [x for x in predicted_mwe_links if x[-1] == "_"]
        gold_strong_mwe_links = [x for x in gold_mwe_links if x[-1] == "_"]
        predicted_mwe_groups = form_groups((x[0], x[1]) for x in
                                           predicted_strong_mwe_links)
        gold_mwe_groups = form_groups((x[0], x[1]) for x in
                                      gold_strong_mwe_links)
        # Label strong MWE groups by their lexcat
        predicted_strong_mwe_links_with_lexcat = [(min(x), max(x)) + (prediction[min(x)].split("-")[1],) for
                                                  x in predicted_mwe_groups]

        gold_strong_mwe_links_with_lexcat = [(min(x), max(x)) + (gold[min(x)].split("-")[1],) for
                                             x in gold_mwe_groups]
        # Get the SWEs with lexcats
        predicted_swe_with_lexcat = []
        gold_swe_with_lexcat = []
        for index, (prediction_word, gold_word) in enumerate(zip(prediction, gold)):
            split_prediction_word = prediction_word.split("-")
            split_gold_word = gold_word.split("-")
            if len(split_prediction_word) > 1 and split_prediction_word[0].lower() == "o":
                predicted_swe_with_lexcat.append((index, index, split_prediction_word[1]))
            if len(split_gold_word) > 1 and split_gold_word[0].lower() == "o":
                gold_swe_with_lexcat.append((index, index, split_gold_word[1]))

        all_lexcats = set(x[-1] for x in (predicted_strong_mwe_links_with_lexcat
                                          + gold_strong_mwe_links_with_lexcat
                                          + predicted_swe_with_lexcat
                                          + gold_swe_with_lexcat))
        # Add up true positives, false positives, false negatives for each lexcat.
        for lexcat in all_lexcats:
            lexcat_predicted_mwes = set([x for x in predicted_strong_mwe_links_with_lexcat if
                                         x[-1] == lexcat])
            lexcat_gold_mwes = set([x for x in gold_strong_mwe_links_with_lexcat if
                                    x[-1] == lexcat])
            lexcat_predicted_swes = set([x for x in predicted_swe_with_lexcat if
                                         x[-1] == lexcat])
            lexcat_gold_swes = set([x for x in gold_swe_with_lexcat if
                                    x[-1] == lexcat])

            true_positives = (len(lexcat_gold_mwes & lexcat_predicted_mwes)
                              + len(lexcat_gold_swes & lexcat_predicted_swes))
            false_negatives = (len(lexcat_gold_mwes - lexcat_predicted_mwes)
                               + len(lexcat_gold_swes - lexcat_predicted_swes))
            false_positives = (len(lexcat_predicted_mwes - lexcat_gold_mwes)
                               + len(lexcat_predicted_swes - lexcat_gold_swes))
            precision_denominator = len(lexcat_predicted_mwes) + len(lexcat_predicted_swes)
            recall_denominator = len(lexcat_gold_mwes) + len(lexcat_gold_swes)
            p_r_counter[lexcat]["true_positives"] += true_positives
            p_r_counter[lexcat]["false_negatives"] += false_negatives
            p_r_counter[lexcat]["false_positives"] += false_positives
            p_r_counter[lexcat]["precision_denominator"] += precision_denominator
            p_r_counter[lexcat]["recall_denominator"] += recall_denominator

    for lexcat in p_r_counter:
        tp = p_r_counter[lexcat]['true_positives']
        precision_denom = p_r_counter[lexcat]['precision_denominator']
        recall_denom = p_r_counter[lexcat]['recall_denominator']
        try:
            precision = tp / precision_denom
        except ZeroDivisionError:
            precision = 0
        try:
            recall = tp / recall_denom
        except ZeroDivisionError:
            recall = 0
        try:
            f1 = 2 * precision * recall / (precision + recall)
        except ZeroDivisionError:
            f1 = 0
        print(f"{lexcat} precision: {100 * precision:.1f} ({tp} / {precision_denom})")
        print(f"{lexcat} recall: {100 * recall:.1f} ({tp} / {recall_denom})")
        print(f"{lexcat} f1: {100 * f1:.1f}")


def parse_mwe_links(mwetags):
    """
    Given a sequence of MWE tags, assert it to be valid,
    and construct links between consecutive MWE elements (strong or weak).
    Every variant of 'I' or 'i' will correspond to one link.
    These links can subsequently be used to form MWE groups: see `form_groups()`
    >>> parse_mwe_links(['O', 'B', 'I_', 'b', 'i~', 'I_', 'B', 'o', 'I_'])
    [(1, 2, '_'), (3, 4, '~'), (2, 5, '_'), (6, 8, '_')]
    >>> parse_mwe_links(['O', 'B', 'I_', 'b', 'i~', 'I~', 'I~', 'o', 'I_'])
    [(1, 2, '_'), (3, 4, '~'), (2, 5, '~'), (5, 6, '~'), (6, 8, '_')]
    >>> parse_mwe_links(['b', 'i_'])
    Traceback (most recent call last):
      ...
    AssertionError: ['b', 'i_']
    >>> parse_mwe_links(['B', 'I~', 'O', 'I~'])
    Traceback (most recent call last):
      ...
    AssertionError: ['B', 'I~', 'O', 'I~']
    >>> parse_mwe_links(['O', 'b', 'i_', 'O'])
    Traceback (most recent call last):
      ...
    AssertionError: ['O', 'b', 'i_', 'O']
    """
    mwetags = [x.split("-")[0] for x in mwetags]
    assert RE_TAGGING.match(''.join(mwetags)), mwetags
    # Sequences such as B I~ O I~ and O b i_ O are invalid.

    # Construct links from BIO tags
    links = []
    last_BI = None
    last_bi = None
    for j, tag in enumerate(mwetags):
        assert tag in STRENGTH

        if tag in {'I','I_','I~'}:
            links.append((last_BI, j, STRENGTH[tag]))
            last_BI = j
        elif tag=='B':
            last_BI = j
        elif tag in {'i','i_','i~'}:
            links.append((last_bi, j, STRENGTH[tag]))
            last_bi = j
        elif tag=='b':
            last_bi = j

    return links


def form_groups(links):
    """
    >>> form_groups([(1, 2), (3, 4), (2, 5), (6, 8), (4, 7)])==[{1,2,5},{3,4,7},{6,8}]
    True
    """
    groups = []
    groupMap = {} # offset -> group containing that offset
    for a,b in links:
        assert a is not None and b is not None,links
        assert b not in groups,'Links not sorted left-to-right: '+repr((a,b))
        if a not in groupMap: # start a new group
            groups.append({a})
            groupMap[a] = groups[-1]
        assert b not in groupMap[a],'Redunant link?: '+repr((a,b))
        groupMap[a].add(b)
        groupMap[b] = groupMap[a]
    return groups


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Given an AllenNLP prediction jsonl file and a gold data file, "
                     "calculate the precision and recall for MWEs per lexcat. "
                     "This differs from the LinkAvg metrics, since P / R / F are "
                     "calculated like they are in NER. Each gold and predicted MWE span"
                     "is an 'entity', and metrics are calculated on these entities"),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--predictions-path", required=True,
                        help="Path to the predictions file.")
    parser.add_argument("--gold-path", required=True,
                        help="Path to the gold data file.")
    args = parser.parse_args()
    main(args.predictions_path, args.gold_path)
