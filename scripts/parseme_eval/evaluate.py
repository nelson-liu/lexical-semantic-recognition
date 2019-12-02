#! /usr/bin/env python3

import argparse
import collections
import itertools
import os
import sys
import tsvlib


parser = argparse.ArgumentParser(description="""
        Evaluate input `prediction` against `gold`.""")
parser.add_argument("--debug", action="store_true",
        help="""Print extra debugging information (you can grep it,
        and should probably pipe it into `less -SR`)""")
parser.add_argument("--combinatorial", action="store_true",
        help="""Run O(n!) algorithm for weighted bipartite matching. 
        You should probably not use this option""")
parser.add_argument("--train", metavar="train_file", dest="train_file",
        required=False, type=argparse.FileType('r'),
        help="""The training file in .cupt format (to calculate 
        statistics regarding seen MWEs)""")
parser.add_argument("--gold", metavar="gold_file", dest="gold_file",
        required=True, type=argparse.FileType('r'),
        help="""The reference gold-standard file in .cupt format""")
parser.add_argument("--pred", metavar="prediction_file", 
        dest="prediction_file", required=True, 
        type=argparse.FileType('r'),
        help="""The system prediction file in .cupt format""")

UNLABELED = '<unlabeled>'
TRAIN_FIELD_NAMES = ['FORM', 'LEMMA']

GOLDPRED_FMT = {
    # gold, pred
    (False, False): "{}",  # normal
    (False,  True): "\x1b[48;5;183m{}\x1b[m",  # pred-only
    (True,  False): "\x1b[48;5;214m{}\x1b[m",  # gold-only
    (True,   True): "\x1b[1;38;5;231;48;5;28m{}\x1b[m", # both (matched!)
}


class Main(object):
    def __init__(self, args):
        sys.excepthook = tsvlib.excepthook
        self.args = args
        # Gold = test.cupt; Pred = test.system.cupt
        if "test.cupt" in self.args.prediction_file.name or "system" in self.args.gold_file.name:
            tsvlib.warn("Something looks wrong in the gold & system arguments.\n" \
                    "Is `{gold_file.name}` really the gold test.cupt file?\n" \
                    "Is `{pred_file.name}` really a system prediction file?",
                    gold_file=self.args.gold_file, pred_file=self.args.prediction_file)


    def run(self):
        if self.args.debug:
            print("DEBUG:  LEGEND:  {} {} {} {}".format(
                GOLDPRED_FMT[(False, False)].format('normal-text'),
                GOLDPRED_FMT[(True, False)].format('gold-only'),
                GOLDPRED_FMT[(False, True)].format('pred-only'),
                GOLDPRED_FMT[(True, True)].format('gold-pred-matched')))
            print("DEBUG:")

        mc_args = dict(debug=self.args.debug, tractable=not self.args.combinatorial)
        self.gold = collections.deque(tsvlib.iter_tsv_sentences(self.args.gold_file))
        self.pred = collections.deque(tsvlib.iter_tsv_sentences(self.args.prediction_file))
        seen = SeenInfo(self.args.train_file)

        base_stats = Statistics(mc_args)
        categ2stats = collections.defaultdict(lambda: Statistics(mc_args))
        continuity2stats = collections.defaultdict(lambda: Statistics(mc_args))
        multitokenness2stats = collections.defaultdict(lambda: Statistics(mc_args))
        field_whetherseen2stats = collections.defaultdict(lambda: Statistics(mc_args))  # dict[(field, bool)] -> stats
        field_variantness2stats = collections.defaultdict(lambda: Statistics(mc_args))  # dict[(field, bool)] -> stats

        while self.gold or self.pred:
            self.check_eof()
            sent_gold = self.gold.popleft()
            sent_pred = self.pred.popleft()
            sent_gold.absorb_mwes_from_contraction_ranges()
            sent_pred.absorb_mwes_from_contraction_ranges()
            if self.args.debug:
                self.print_debug_pairing(sent_gold, sent_pred)
            self.compare_sentences(sent_gold, sent_pred)
            categories = self.mwe_categs(sent_gold) | self.mwe_categs(sent_pred)
            mweinfos_gold = sent_gold.mwe_infos().values()
            mweinfos_pred = sent_pred.mwe_infos().values()

            self.add_to_stats(
                sent_gold, base_stats, mweinfos_gold, mweinfos_pred,
                debug_header="Global:")

            for category in list(sorted(categories, key=str)):
                g = self.mweinfos_per_categ(mweinfos_gold, category)
                p = self.mweinfos_per_categ(mweinfos_pred, category)
                self.add_to_stats(sent_gold, categ2stats[category], g, p,
                    debug_header="Category {}:".format(category or UNLABELED))

            for continuity in [True, False]:
                g = self.mweinfo_per_continuity(mweinfos_gold, continuity)
                p = self.mweinfo_per_continuity(mweinfos_pred, continuity)
                self.add_to_stats(sent_gold, continuity2stats[continuity], g, p,
                    debug_header="Continuous:" if continuity else "Discontinuous:")

            for multitokenness in [True, False]:
                g = self.mweinfo_per_multitokenness(mweinfos_gold, multitokenness)
                p = self.mweinfo_per_multitokenness(mweinfos_pred, multitokenness)
                self.add_to_stats(sent_gold, multitokenness2stats[multitokenness], g, p,
                    debug_header="{}-token:".format("Multi" if multitokenness else "Single"))

            if self.args.train_file:
                for whetherseen in [True, False]:
                    g = seen.mweinfo_per_whetherseen(mweinfos_gold, "LEMMA", whetherseen)
                    p = seen.mweinfo_per_whetherseen(mweinfos_pred, "LEMMA", whetherseen)
                    self.add_to_stats(sent_gold, field_whetherseen2stats[("LEMMA", whetherseen)], g, p,
                        debug_header="{}-in-train:".format("Seen" if whetherseen else "Unseen"))

                for variantness in [True, False]:
                    # We interpret variantness==False as "MWEs that were seen and are identical"
                    g = seen.mweinfo_per_variantness(mweinfos_gold, "LEMMA", "FORM", variantness)
                    p = seen.mweinfo_per_variantness(mweinfos_pred, "LEMMA", "FORM", variantness)
                    self.add_to_stats(sent_gold, field_variantness2stats[("LEMMA", "FORM", variantness)], g, p,
                        debug_header="{}-train:".format("Variant-of" if variantness else "Identical-to"))

            if self.args.debug:
                print("DEBUG:")


        #------------------------------------------
        print("## Global evaluation")
        base_stats.print_stats(prefix='')
        print()

        print("## Per-category evaluation (partition of Global)")
        for category in sorted(categ2stats, key=str):
            prefix = '{}: '.format(category or UNLABELED)
            categ2stats[category].print_mwebased_proportion(
                prefix, baseline=base_stats)
            categ2stats[category].print_stats(prefix)
        print()

        print("## MWE continuity (partition of Global)")
        for continuity in [True, False]:
            prefix = "Continuous: " if continuity else "Discontinuous: "
            continuity2stats[continuity].print_mwebased_proportion(
                        prefix, baseline=base_stats)
            continuity2stats[continuity].c_mwebased.print_p_r_f(prefix)
        print()

        print("## Number of tokens (partition of Global)")
        for multitokenness in [True, False]:
            prefix = "{}-token: ".format("Multi" if multitokenness else "Single")
            multitokenness2stats[multitokenness].print_mwebased_proportion(
                        prefix, baseline=base_stats)
            multitokenness2stats[multitokenness].c_mwebased.print_p_r_f(prefix)
        print()

        if self.args.train_file:
            if not seen.mwe_fieldindex_sets["LEMMA"]:
                tsvlib.warn("found no MWEs in training file (in field={field_name})",
                            field_name="LEMMA", position='')

            else:
                print("## Whether seen in train (partition of Global)")
                for whetherseen in [True, False]:
                    prefix = "{}-in-train: ".format("Seen" if whetherseen else "Unseen")
                    field_whetherseen2stats[("LEMMA", whetherseen)] \
                        .print_mwebased_proportion(prefix, baseline=base_stats)
                    field_whetherseen2stats[("LEMMA", whetherseen)].c_mwebased.print_p_r_f(prefix)
                print()

                print("## Whether identical to train (partition of Seen-in-train)")
                for variantness in [True, False]:
                    prefix = "{}-train: ".format("Variant-of" if variantness else "Identical-to")
                    field_variantness2stats[("LEMMA", "FORM", variantness)] \
                        .print_mwebased_proportion(prefix, baseline=field_whetherseen2stats[("LEMMA", True)])
                    field_variantness2stats[("LEMMA", "FORM", variantness)].c_mwebased.print_p_r_f(prefix)
                print()


    #=============================================================
    def add_to_stats(self, sent_gold: tsvlib.TSVSentence,
                     target_stats: 'Statistics', mweinfos_gold: list,
                     mweinfos_pred: list, *, debug_header="???"):
        r'''Add statistics about mweinfos to `target_stats`.'''
        g_tokensets = self.to_tokensets(mweinfos_gold)
        p_tokensets = self.to_tokensets(mweinfos_pred)
        if self.args.debug:
            self.print_debug_tokensets(debug_header, "gold", g_tokensets)
            self.print_debug_tokensets(debug_header, "pred", p_tokensets)
        inc_mwe, inc_tok = target_stats.increment_stats(g_tokensets, p_tokensets)
        if self.args.debug:
            target_stats.c_mwebased.print_debug(sent_gold, inc_mwe, g_tokensets, p_tokensets, debug_header)
            target_stats.c_tokbased.print_debug(sent_gold, inc_tok, g_tokensets, p_tokensets, debug_header)
            print("DEBUG: +------------------------------------")


    def print_debug_pairing(self, g, p):
        triples = [["ID", "GOLD", "PRED", "WORD"]]
        for i, tok_g in enumerate(g.words):
            try:
                tok_p_mwe_codes = p.words[i].mwe_codes()
            except IndexError:
                tok_p_mwe_codes = ""

            triples.append(["t{}".format(i+1),
                    ";".join(sorted(tok_g.mwe_codes())),
                    ";".join(sorted(tok_p_mwe_codes)),
                    tok_g.get('FORM', '_')])
        print("DEBUG: +============================================================")
        for triple in triples:
            triple[0] = "{:<10}".format(triple[0])
            triple[1:3] = ["{:<17}".format(x) for x in triple[1:3]]
            print("DEBUG: |", "  ".join(triple), sep="")
        print("DEBUG: +============================================================")

    def print_debug_tokensets(self, debug_header, name, mwes):
        print("DEBUG: | {} {} = {}".format(debug_header, name, mwes2t(mwes)))


    def mwe_categs(self, sent: tsvlib.TSVSentence) -> set:
        r'''Get the set of MWE categories referenced in sentence.'''
        return set(mweinfo.category for mweinfo in sent.mwe_infos().values())


    def mweinfos_per_categ(self, mweinfos: list, categ: str):
        r'''Return a sublist of MWEInfo instances for given category.'''
        return [m for m in mweinfos if m.category == categ]

    def mweinfo_per_continuity(self, mweinfos: list, continuity: bool):
        r'''Return a sublist of MWEInfo instances that are (dis)continuous.'''
        return [m for m in mweinfos if continuity == (m.n_gaps() == 0)]

    def mweinfo_per_multitokenness(self, mweinfos: list, multitoken: bool):
        r'''Return a sublist of MWEInfo instances that are {multi,single}-token.'''
        return [m for m in mweinfos if multitoken == (m.n_tokens() >= 2)]

    def to_tokensets(self, mweinfos: list):
        r"""Return a list of MWEs, each one represented as a set of integers.
        MWEs are ordered in an "arbitary" order (but in practice, we sort it, for human readability).
        NOTE: we group identical MWEs as a single unit, as per the Shared Task meeting's discussion
        regarding a set-based definition of MWEs.
        """
        tokensets = set(frozenset(i+1 for i in m.token_indexes) for m in mweinfos)
        return list(sorted(tokensets, key=lambda tokenset: list(sorted(tokenset))))


    def check_eof(self):
        r"""Generate an error if one of the files is in EOF and the other is not."""
        if not self.gold:
            error("Prediction file is larger than the gold file (extra data?)")
        if not self.pred:
            error("Prediction file is smaller than the gold file (missing data?)")

    def compare_sentences(self, sent_g, sent_p):
        r'''Warn if sentence sizes do not match.'''
        if len(sent_g.words) != len(sent_p.words):
            tsvlib.global_last_lineno(None, 0)
            len_g, len_p = len(sent_g.words), len(sent_p.words)
            tsvlib.warn(
                "Sentence sizes do not match\n" \
                "In sentence starting at `{args.gold_file.name}` "\
                "line {g.lineno_beg} ({len_g} tokens)\n" \
                "In sentence starting at `{args.prediction_file.name}` "\
                "line {p.lineno_beg} ({len_p} tokens)",
                g=sent_g, p=sent_p, args=self.args, len_g=len_g, len_p=len_p)


class SeenInfo:
    r'''Encapsulates the handling of whether an MWE has been seen in train.'''
    def __init__(self, train_file):
        self.train_file = train_file
        if self.train_file is not None:
            self.mwe_fieldindex_sets, self.mwe_field_sets, self.mwe_spans = {}, {}, {}
            sents = list(tsvlib.iter_tsv_sentences(self.train_file))
            for field_name in TRAIN_FIELD_NAMES:
                self.mwe_fieldindex_sets[field_name] = self._calc_mwe_fieldindex_sets(field_name, sents)
                self.mwe_field_sets[field_name] = self._calc_mwe_field_sets(field_name)
                self.mwe_spans[field_name] = frozenset(
                    span for sent in sents for span in sent.iter_mwe_fields_including_span(field_name))

    def _calc_mwe_fieldindex_sets(self, field_name: str, sents: list):
        r'''Return a Set[Frozenset[(field_value: str, index: int)]]
        The indexes can be used to check exact MWE matches (with no reordering).
        Example for field_name=="LEMMA":
          {{(take, 1), (bath, 3)}, {(there, 1), (be, 2)}}
        '''
        return set(fieldindex_set for sentence in sents for fieldindex_set
                   in sentence.iter_mwe_fields_and_normalizedindexes(field_name))

    def _calc_mwe_field_sets(self, field_name: str):
        r'''Return a Set[Counter[field_value: str]]
        Example for field_name=="LEMMA":
          {{take, bath}, {look, up}, {there, be}, {to(2x), be(2x), or, not}}
        '''
        return set(
            tsvlib.FrozenCounter(field for (field, index) in fieldindex_set)
            for fieldindex_set in self.mwe_fieldindex_sets[field_name])

    def mweinfo_per_whetherseen(self, mweinfos: list, field_name: str, whetherseen: bool):
        r'''Return a sublist of MWEInfo instances that are {multi,single}-token.'''
        return [m for m in mweinfos if whetherseen == self._seen_in_train(m, field_name)]

    def _seen_in_train(self, mweinfo: tsvlib.MWEInfo, field_name: str):
        r'''Return True iff `mweinfo` was seen in train.'''
        field_counter = tsvlib.FrozenCounter(
            mweinfo.sentence.words[i].get(field_name, "_") for i in mweinfo.token_indexes)
        return field_counter in self.mwe_field_sets[field_name]

    def mweinfo_per_variantness(self, mweinfos: list, seen_field_name: str, variant_field_name: str, variantness: bool):
        r'''Return a sublist of MWEInfo instances that
        are (non-)exact variants of MWEs seen in train.
        '''
        return [m for m in mweinfos if self._variant_of_train(m, seen_field_name, variant_field_name, variantness)]

    def _variant_of_train(self, mweinfo: tsvlib.MWEInfo, seen_field_name: str, variant_field_name: str, variantness: bool):
        r'''Return True iff `mweinfo` is a (non-)exact variant of an MWE seen in train.'''
        if not self._seen_in_train(mweinfo, seen_field_name):
            return False  # Can only be a (non-)exact variant if seen in train
        field_with_span = mweinfo.field_including_span(variant_field_name, "FORM")
        return variantness != (field_with_span in self.mwe_spans[variant_field_name])




#################################################
def mwe2t(mwe):
    r"""mwe2t(frozenset[int]) -> str
    Return a string representation such as "t1_t3_t4"."""
    return "_".join("t{}".format(i) for i in sorted(mwe))

def mwes2t(mwes):
    r"""mwes2t(frozenset[frozenset[int]]) -> str"""
    return "{" + ", ".join(sorted(mwe2t(mwe) for mwe in mwes)) + "}"

def pairing2t(pairing):
    r"""pairing2t(dict[frozenset[int], frozenset[int]]) -> str"""
    return "{" + ", ".join(sorted("{}=>{}".format(mwe2t(mwe1), mwe2t(mwe2))
            for (mwe1,mwe2) in pairing.items())) + "}"


def error(message, **kwargs):
    r"""Print error message and quit."""
    tsvlib.warn(message, warntype="ERROR", **kwargs)
    sys.exit(1)



#################################################
Increment = collections.namedtuple('Increment', 'plus_g plus_p plus_correct pairing')  # (int, int, int, dict)


class MatchCounter:
    r'''Counts P/R/F statistics.'''
    def __init__(self, name, debug, tractable):
        self.name = name
        self.debug = debug
        self.tractable = tractable
        self.total_gold = 0  # recall = correct/total_gold
        self.total_pred = 0  # precision = correct/total_pred
        self.correct = 0

    def print_debug(self, sent: tsvlib.TSVSentence, inc: Increment,
                    g_tokensets: list, p_tokensets: list, header: str):
        print("DEBUG: | {header} Mapping gold=>pred ({name}): {debug_pairing}  " \
                "@@ P+={plus_correct}/{plus_p} R+={plus_correct}/{plus_g}".format(
                header=header, name=self.name, debug_pairing=pairing2t(inc.pairing),
                plus_correct=inc.plus_correct, plus_g=inc.plus_g, plus_p=inc.plus_p))

        for map_gold, map_pred in inc.pairing.items():
            sentence = " ".join(GOLDPRED_FMT[(i in map_gold, i in map_pred)]
                                .format(tok.get('FORM', '_')) for i, tok in enumerate(sent.words, 1))
            print("DEBUG: | {header} => MATCH gold/pred ({name}): {sentence}" \
                  .format(header=header, name=self.name, sentence=sentence))
        
        for subname, pairs in [("gold", {(g, frozenset()) for g in set(g_tokensets)-inc.pairing.keys()}),
                               ("pred", {(frozenset(), p) for p in set(p_tokensets)-set(inc.pairing.values())})]:
            for g_ts, p_ts in pairs:
                sentence = " ".join(GOLDPRED_FMT[(i in g_ts, i in p_ts)]
                                    .format(tok.get('FORM', '_')) for i, tok in enumerate(sent.words, 1))
                print("DEBUG: | {header} => FAIL: ONLY {subname} ({name}): {sentence}" \
                      .format(subname=subname, header=header, name=self.name, sentence=sentence))


    def increment(self, inc: Increment):
        r'''(Low-level helper method to increment counters for precision and recall).'''
        self.total_gold += inc.plus_g
        self.total_pred += inc.plus_p
        self.correct += inc.plus_correct
        return inc


    def increment_mwebased(self, g_tokensets: list, p_tokensets: list):
        r'''Pair up entries between g_tokensets and p_tokensets, and increment counters (per MWE).'''
        pairing = {x:x for x in set(g_tokensets) & set(p_tokensets)}
        inc = Increment(len(g_tokensets), len(p_tokensets), len(pairing), pairing)
        return self.increment(inc)


    def increment_tokbased(self, g_tokensets: list, p_tokensets: list):
        r'''Pair up entries between g_tokensets and p_tokensets, and increment counters (per token).'''
        pairing = tokbased_pairing(g_tokensets, p_tokensets, tractable=self.tractable)
        inc = Increment(sum(len(m) for m in g_tokensets), sum(len(m) for m in p_tokensets),
                        sum(len(a&b) for (a, b) in pairing.items()), pairing)
        return self.increment(inc)


    def print_p_r_f(self, prefix: str):
        r'''Prints P/R/F'''
        precision = self.correct / (self.total_pred or 1)
        recall = self.correct / (self.total_gold or 1)
        f1 = 2*precision*recall/(precision+recall) if precision else 0
        print("* {prefix}{self.name}: " \
              "P={self.correct}/{self.total_pred}={precision:.4f} " \
              "R={self.correct}/{self.total_gold}={recall:.4f} " \
              "F={f1:.4f}".format(self=self, precision=precision,
                                    recall=recall, f1=f1, prefix=prefix))


class Statistics:
    r'''Counts P/R/F statistics for both MWE-based and Token-based measures.'''
    def __init__(self, matchcounter_kwargs: dict):
        self.c_mwebased = MatchCounter("MWE-based", **matchcounter_kwargs)    # exact match
        self.c_tokbased = MatchCounter("Tok-based", **matchcounter_kwargs)  # fuzzy match

    def increment_stats(self, g_tokensets: list, p_tokensets: list) -> (Increment, Increment):
        r'''Increment P/R/F statistics for both MWE-based and Token-based measures.'''
        inc_mwe = self.c_mwebased.increment_mwebased(g_tokensets, p_tokensets)
        inc_tok = self.c_tokbased.increment_tokbased(g_tokensets, p_tokensets)
        return inc_mwe, inc_tok

    def print_stats(self, prefix: str):
        r'''Prints P/R/F for {mwe,tok}based'''
        self.c_mwebased.print_p_r_f(prefix)
        self.c_tokbased.print_p_r_f(prefix)

    def print_mwebased_proportion(self, prefix: str, baseline: 'Statistics'):
        r'''Print proportion of self/total.'''
        n_gold_self = self.c_mwebased.total_gold
        n_gold_baseline = baseline.c_mwebased.total_gold
        n_pred_self = self.c_mwebased.total_pred
        n_pred_baseline = baseline.c_mwebased.total_pred
        print("* {}MWE-proportion: gold={}/{}={:.0%} pred={}/{}={:.0%}".format(prefix,
              n_gold_self, n_gold_baseline, n_gold_self/(n_gold_baseline or 1),
              n_pred_self, n_pred_baseline, n_pred_self/(n_pred_baseline or 1)))



##################################################
def tokbased_pairing(g_tokensets: list, p_tokensets: list, tractable: bool) -> dict:
    r'''Return a dict representing a pairing of entries between g_tokensets
    and p_tokensets.  The key is a frozenset[int] (gold token indexes).
    The value is a frozenset[int] (prediction token indexes).
    This dict maximizes the number of tokens in common between gold & pred.
    '''
    if tractable:  # Use O(n^3) algorithm
        if not g_tokensets or not p_tokensets: return {}
        return ParsemeBipartiteGraph(g_tokensets, p_tokensets).mapping

    else:  # Use O(n!) algorithm
        g_tokensets += [frozenset()] * (len(p_tokensets) - len(g_tokensets))
        p_tokensets += [frozenset()] * (len(g_tokensets) - len(p_tokensets))
        best, best_count = {}, 0
        for p_tokenset_permut in itertools.permutations(p_tokensets):
            pairing = {a:b for (a,b) in zip(g_tokensets, p_tokenset_permut) if a and b}
            pairing_count = sum(len(set(a)&set(b)) for (a,b) in pairing.items())
            if pairing_count > best_count:
                best, best_count = pairing, pairing_count
        return best


class ParsemeBipartiteGraph:
    def __init__(self, g_tokensets: list, p_tokensets: list):
        from bmc_munkres import munkres
        cost_mtx = [[self.cost(g, p) for p in p_tokensets] for g in g_tokensets]
        result_index_pairs = munkres.Munkres().compute(cost_mtx)
        self.mapping = {g_tokensets[i_g]: p_tokensets[i_p]
                        for (i_g, i_p) in result_index_pairs}

    def cost(self, g: set, p: set) -> int:
        return - self.weight(g, p)
    
    def weight(self, g: set, p: set) -> int:
        return len(g & p)  # edge weight = number of MWE tokens in common



#####################################################

if __name__ == "__main__":
    Main(parser.parse_args()).run()
