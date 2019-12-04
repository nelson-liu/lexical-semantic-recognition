import os
import tempfile
from collections import defaultdict, Counter
from typing import Dict, List, TextIO

from UDlextag2json import load_sents as unpack_sents
from allennlp.training.metrics.metric import Metric
from conllulex2json import load_sents, print_json
from overrides import overrides
from streuseval import eval_sys
from supersenses import coarsen_pss

DEPTH = 4


@Metric.register("streuseval")
class Streuseval(Metric):
    def __init__(self):
        self._scores = defaultdict(lambda: defaultdict(Counter))

    @overrides
    def __call__(self,
                 batch_tags: List[List[str]],
                 batch_gold_tags: List[List[str]],
                 batch_upos: List[List[str]]):
        tempdir = tempfile.mkdtemp()
        gold_path = os.path.join(tempdir, "gold.json")
        predicted_path = os.path.join(tempdir, "predicted.autoid.json")
        # TODO(danielhers): Unused variable:
        # unpacked_predicted_path = os.path.join(tempdir, "unpacked_predicted.autoid.json")
        with open(predicted_path, "w", encoding="utf-8") as predicted_file, \
                open(gold_path, "w", encoding="utf-8") as gold_file:
            for tags, gold_tags, upos in zip(
                    batch_tags,
                    batch_gold_tags,
                    batch_upos):
                write_conllulex_formatted_tags_to_file(predicted_file,
                                                       gold_file,
                                                       tags,
                                                       gold_tags,
                                                       upos)
        with open(predicted_path, encoding="utf-8") as predicted_file:
            # TODO(danielhers): Unused variable:
            # \ open(unpacked_predicted_path, "w", encoding="utf-8") as unpacked_predicted_file:
            print_json(unpack_sents(predicted_file))
        with open(gold_path, encoding="utf-8") as gold_file:
            # TODO(danielhers): Unused variable:
            # \ open(unpacked_predicted_path, encoding="utf-8") as unpacked_predicted_file:
            gold_sents = list(load_sents(gold_file, ss_mapper=ss_mapper))
            self._scores = eval_sys(predicted_file, gold_sents, ss_mapper)  # TODO accumulate

    @overrides
    def get_metric(self,
                   reset: bool = False) -> Dict[str, float]:
        return self._scores

    @overrides
    def reset(self):
        self._scores = defaultdict(lambda: defaultdict(Counter))


def ss_mapper(supersense):
    return coarsen_pss(supersense, DEPTH) if supersense.startswith('p.') else supersense


def write_conllulex_formatted_tags_to_file(prediction_file: TextIO,
                                           gold_file: TextIO,
                                           batch_tags: List[str],
                                           batch_gold_tags: List[str],
                                           batch_upos: List[str]):
    """
    Prints predicate argument predictions and gold labels for a single sentence
    to two provided file references.
    The CoNLL-U-Lex format is described in
    `the STREUSLE documentation <https://github.com/nert-nlp/streusle/blob/master/CONLLULEX.md>`_ .
    Parameters
    ----------
    prediction_file : TextIO, required.
        A file reference to print predictions to.
    gold_file : TextIO, required.
        A file reference to print gold labels to.
    batch_gold_tags : List[str], required.
        The predicted tags.
    batch_gold_tags : List[str], required.
        The gold tags.
    batch_upos : List[str], required.
        The UPOS tags.
    """
    for predicted, gold, upos in zip(batch_tags,
                                     batch_gold_tags,
                                     batch_upos):
        # TODO add metadata: sent_id, text, streusle_sent_id, mwe
        # TODO add UD columns: ID, FORM, LEMMA, XPOS, FEATS, HEADS, DEPREL, DEPS, MISC
        # TODO add lex columns: SMWE, LEXCAT, LEXLEMMA, SS, SS2, WMWE, WCAT, WLEMMA
        print(upos, predicted, sep="\t", file=prediction_file)
        print(upos, gold, sep="\t", file=gold_file)
    print(file=prediction_file)
    print(file=gold_file)
