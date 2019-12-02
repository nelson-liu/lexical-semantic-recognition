#! /usr/bin/env python3

import argparse
import collections
import math
import os
import re
import sys

import tsvlib


RE_STATLINE = re.compile(r'([^=]*) ([^= ]+=.*)')  # to parse (prefix, statline)
RE_STAT = re.compile(r'\b(\S*?)=((\d*?)/(\d*?)=)?([\d.]*)')  # e.g. P=3/6=0.5000

Stat = collections.namedtuple('Stat', 'statname num_denom numerator denominator score')
StatLine = collections.namedtuple('StatLine', 'type prefix stats')   # (str, str, List[Stat])

BRACES = ('\x1b[36m{}\x1b[0m' if tsvlib.COLOR_STDOUT else '{}')
FRACT = ('\x1b[37m{}\x1b[0m' if tsvlib.COLOR_STDOUT else '{}')


parser = argparse.ArgumentParser(description="""
        Calculate macro-average of scores from multiple evaluate.py output files.""")
parser.add_argument("--operation", choices=('avg', 'avg+stddev', 'list'), default='avg+stddev',
        help="""Type of output (default: "avg+stddev")""")
parser.add_argument("evaluation_files", nargs='+', type=argparse.FileType('r'),
        help="""Files that were produced as output of evaluate.py""")


def format_float(value: float):
    r"""Return pretty-formatted float."""
    return "{:.4f}".format(value)


class Main(object):
    def __init__(self, args):
        self.args = args

    def run(self):
        title2blocks = collections.OrderedDict()
        for eval_file in self.args.evaluation_files:
            blocks = parse_blocks(eval_file)
            for block in blocks:
                title2blocks.setdefault(block.title, []).append(block)

        for title, all_blocks in title2blocks.items():
            print(title)
            prefixes = list(uniq(p for block in all_blocks for p in block.prefix2statline.keys()))

            for prefix in prefixes:
                parallel_statlines = [b.prefix2statline[prefix] for b in all_blocks
                                      if prefix in b.prefix2statline]
                strtype = parallel_statlines[0].type
                prefix = parallel_statlines[0].prefix
                merged = self.merge_statlines(parallel_statlines)

                if strtype == 'PRF':
                    if self.args.operation in ['avg', 'avg+stddev']:
                        # Re-calculate F1 based on precision and recall
                        prec, recall = [float(v.split('=')[-1].split('(')[0]) for v in merged[:2]]
                        merged[-1] = format_float((2*prec*recall)/((prec+recall) or 1))

                statnames = [s.statname for s in parallel_statlines[0].stats]
                pairs = zip(statnames, merged)
                fmt = "{}={}" if strtype == "PRF" else "{}={}%"
                averages = " ".join(fmt.format(p[0], BRACES.format(p[1])) for p in pairs)
                print(prefix, averages, FRACT.format(" @({}/{})".format(len(parallel_statlines), len(all_blocks))))
            print()


    def merge_statlines(self, statlines: list) -> list:
        r"""Return a List[str] for each statname, merging data from all `statlines`."""
        types = [(p_line.type, p_line.prefix) for p_line in statlines]
        if any(t != types[0] for t in types):
            exit("ERROR: non-matching parallel lines: prefix={!r} types={}".format(statlines[0].prefix, types))

        statlist_per_line = [statline.stats for statline in statlines]
        statlist_per_statname = list(zip(*statlist_per_line))
        return [self.calc_average_str(statlist) for statlist in statlist_per_statname]


    def calc_average_str(self, stats: list):
        scores = [float(stat.score) for stat in stats]
        if self.args.operation == 'avg':
            avg = sum(scores) / len(scores)
            return format_float(avg)
        elif self.args.operation == 'avg+stddev':
            avg = sum(scores) / len(scores)
            sumsq = sum((s-avg)**2 for s in scores)
            # Bessel's N-1 correction for sample stddev:
            stddev = "inf"
            if len(scores) > 1:
                stddev = format_float(math.sqrt(sumsq / (len(scores)-1 or 1)))
            return "{}(±{})".format(format_float(avg), stddev)
        elif self.args.operation == 'list':
            return "[" + ",".join(format_float(s) for s in scores) + "]"
        else:
            assert False


class Block:
    r"""Represents a block such as:
    | ## Global evaluation          <-- title
    | * MWE-based: P=485/496=0.9778 R=485/501=0.9681 F=0.9729       <-- prefix2statline["* MWE-based:"]
    | * Tok-based: P=1073/1087=0.9871 R=1073/1087=0.9871 F=0.9871   <-- prefix2statline["* Tok-based:"]
    """
    def __init__(self, text: str):
        if not "\n" in text:
            exit("ERROR: Empty block: {!r}".format(text))

        self.title, rest = text.split("\n", 1)
        if not self.title.startswith('#'):
            exit('Block title does not start with "#": {!r}'.format(self.title))

        statlines = [parse_statline(x) for x in rest.split("\n") if x]
        self.prefix2statline = collections.OrderedDict(
            (sl.prefix, sl) for sl in statlines)  # type: dict[str, StatLine]

    def __repr__(self):
        return "Block<{!r}>".format(self.title)


def parse_blocks(fileobj):
    r"""Split text data in fileobj into objects of type Block"""
    ret = fileobj.read().split("\n\n")
    return [Block(x) for x in ret if x]


def parse_statline(line: str):
    r"""Return an instance of StatLine or None."""
    if not line.strip() or line.startswith('#'):
        exit("ERROR: Unexpected line: {!r}".format(line))

    prefix, rest = RE_STATLINE.match(line).groups()
    stats = [Stat(*x) for x in RE_STAT.findall(rest)]
    statnames = [s.statname for s in stats]

    if statnames == ['P', 'R', 'F']:
        return StatLine('PRF', prefix, stats)
    if statnames == ['gold', 'pred']:
        return StatLine('GOLDPRED', prefix, stats)
    exit("ERROR: Bad line: {!r} {!r}".format(line, statnames))


def uniq(things: list):
    r"""Yield the first occurrence of each element."""
    seen = set()
    for thing in things:
        if not thing in seen:
            yield thing
        seen.add(thing)


#####################################################

if __name__ == "__main__":
    Main(parser.parse_args()).run()
