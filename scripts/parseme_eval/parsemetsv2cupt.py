#! /usr/bin/env python3

import argparse
import re
import sys

parser = argparse.ArgumentParser(description="""
        Convert input file format from PARSEME-TSV (edition 1.0)
        to the CUPT format (edition 1.1).""")
parser.add_argument("--underspecified-mwes", action='store_true',
        help="""If set, represent empty PARSEME:MWE slots as "_" instead of "*" (for blind).""")
parser.add_argument("--input", type=argparse.FileType('r'), required=True,
        help="""Path to input file (in FoLiA XML or PARSEME TSV format)""")


class Main:
    def __init__(self, args):
        self.args = args

    def run(self):
        missing_mwe_annot = "_" if self.args.underspecified_mwes else "*"
        print('# global.columns = ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC PARSEME:MWE')
        print('#')
        print('#')

        for sentno, block_text in enumerate(self.iter_blocks(), 1):
            if "# sent_id = " in block_text:
                block_text = block_text.replace('# sent_id = ', '# source_sent_id = . . ')
            if not "# source_sent_id =" in block_text:
                print("# source_sent_id = . . dummy-{}".format(sentno))

            if not "# text =" in block_text:
                print("# text = dummy")

            for line in block_text.split("\n"):
                if line.startswith('#'):
                    print(line)
                    continue
                try:
                    rank, form, nsp, mwe = line.split("\t")
                    nsp = "SpaceAfter=No" if nsp != "_" else "_"
                    mwe = missing_mwe_annot if mwe == "_" else mwe
                    print(rank, form, *(["_"]*7+[nsp, mwe]), sep="\t")
                except Exception as e:
                    print("====> ERROR IN LINE: {!r}".format(line), file=sys.stderr)
                    raise
            print()

    def iter_blocks(self):
        r"""Yield blocks (each block being a sentence in PARSEME-TSV format)"""
        blocks = re.split(r'\n{2,}', self.args.input.read())
        return (b for b in blocks if b)


#####################################################

if __name__ == "__main__":
    Main(parser.parse_args()).run()
