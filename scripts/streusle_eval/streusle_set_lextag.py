import ast
import json
import logging

import argparse
from itertools import repeat

from conllulex2json import load_sents, print_json
from supersenses import coarsen_pss

logger = logging.getLogger(__name__)


class SSMapper:
    def __init__(self, depth):
        self.depth = depth

    def __call__(self, ss):
        return coarsen_pss(ss, self.depth) if ss.startswith('p.') else ss


def swap_tags(sents, preds):
    for sent, pred in zip(sents, preds):
        for tok, lextag, upos in zip(sent["toks"], pred["tags"], pred.get("upos_tags", repeat(None))):
            tok["lextag"] = lextag
            if upos is not None:
                tok["upos"] = upos
        yield sent


def load_tags(lines):
    for line in lines:
        try:
            yield json.loads(line)
        except json.decoder.JSONDecodeError:
            yield ast.literal_eval(line)


def main(args):
    with open(args.fname, encoding="utf-8") as f, open(args.lextags, encoding="utf-8") as tags_lines:
        sents = load_sents(f, ss_mapper=SSMapper(args.depth), validate_type=False, validate_pos=False)
        preds = load_tags(tags_lines)
        print_json(swap_tags(sents, preds))


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(levelname)s "
                               "- %(name)s - %(message)s",
                        level=logging.INFO)
    argparser = argparse.ArgumentParser(description="Swap lextags into a STREUSLE file")
    argparser.add_argument("fname", help="conllulex or json file with full STREUSLE annotation")
    argparser.add_argument("lextags", help="jsonlines file: each line's 'tags' entry is a sentence's list of lextags"
                                           "in serialized list notation, e.g., ['B-ADV', 'I~-V-v.communication']")
    argparser.add_argument('--depth', metavar='D', type=int, choices=range(1, 5), default=4,
                           help='depth of hierarchy at which to cluster SNACS supersense labels '
                                '(default: 4, i.e. no collapsing)')
    main(argparser.parse_args())
