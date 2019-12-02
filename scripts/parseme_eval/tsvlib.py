#! /usr/bin/env python3

r"""
    This is a small library for reading and interpreting
    the new ConLLU-PLUS format.

    This format allows any column from CoNLLU (e.g. ID, FORM...)
    As in CoNLL-U, empty columns are represented by "_".

    The first line of these files should have the form:
    # global.columns = ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC PARSEME:MWE

    The column "PARSEME:MWE" can be used to indicate
    MWE codes (e.g. "3:LVC.full;2;5:VID") or be EMPTY.
"""


import collections
import os
import sys

UNDERSP = "_"
SINGLEWORD = "*"

CONLLUP_FIELDS = 'ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC PARSEME:MWE'.split()


#######################################
def interpret_color_request(stream, color_req: str) -> bool:
    r"""Interpret environment variables COLOR_STDOUT and COLOR_STDERR ("always/never/auto")."""
    return color_req == 'always' or (color_req == 'auto' and stream.isatty())

# Flags indicating whether we want to use colors when writing to stderr/stdout
COLOR_STDOUT = interpret_color_request(sys.stdout, os.getenv('COLOR_STDOUT', 'auto'))
COLOR_STDERR = interpret_color_request(sys.stderr, os.getenv('COLOR_STDERR', 'auto'))


############################################################
class TSVSentence:
    r"""A list of TSVTokens.
        TSVTokens may include ranges and sub-tokens.

        For example, if we have these TSVTokens:
            1   You
            2-3 didn't   -- a range
            2   did      -- a sub-token
            3   not      -- a sub-token
            4   go
        Iterating through `self.words` will yield ["You", "did", "not", "go"].
        You can access the range ["didn't"] through `self.contractions`.
    """
    def __init__(self, filename, lineno_beg, words=None, contractions=None):
        self.filename = filename
        self.lineno_beg = lineno_beg
        self.words = words or []
        self.contractions = contractions or []

    def __str__(self):
        return "TSVSentence({!r}, {!r}, {!r}, {!r})".format(self.filename,
                self.lineno_beg, self.words, self.contractions)

    def append(self, token):
        r"""Add `token` to either `self.words` or `self.contractions`."""
        L = (self.contractions if token.is_contraction() else self.words)
        L.append(token)

    def subtoken_indexes(self):
        r"""Return a set with the index of every sub-word."""
        sub_indexes = set()
        for token in self.contractions:
            sub_indexes.update(token.contraction_range())
        return sub_indexes

    def iter_words_and_ranges(self):
        r"""Yield all tokens, including ranges.
        For example, this function may yield ["You", "didn't", "did", "not", "go"].
        """
        index2contractions = collections.defaultdict(list)
        for c in self.contractions:
            index2contractions[c.contraction_range().start].append(c)
        for i, token in enumerate(self.words):
            for c in index2contractions[i]:
                yield c
            yield token

    def mwe_infos(self):
        r"""Return a dict {mwe_id: MWEInfo} for all MWEs in this sentence."""
        mwe_infos = {}
        for token_index, token in enumerate(self.words):
            global_last_lineno(self.filename, token.lineno)
            for mwe_id, mwe_categ in token.mwes_id_categ():
                mwe_info = mwe_infos.setdefault(mwe_id, MWEInfo(self, mwe_categ, []))
                if token_index in mwe_info.token_indexes:
                    warn("Ignoring repeated mwe_id ({mwe_id}) in PARSEME:MWE field", mwe_id=mwe_id)
                else:
                    mwe_info.token_indexes.append(token_index)
        return mwe_infos

    def absorb_mwes_from_contraction_ranges(self):
        r"""If a range is part of an MWE, add its subtokens as part of it as well."""
        for c in self.contractions:
            mustwarn = False # warning appears once per contraction...
            for i_subtoken in c.contraction_range():
                more_codes = c.mwe_codes()
                if more_codes:            
                    mustwarn = True # ...not once per contraction element
                    all_codes = self.words[i_subtoken].mwe_codes()
                    all_codes.update(more_codes)
                    # If e.g. 3:IAV and 3 in all_codes, remove 3 and keep only 3:IAV
                    for code in sorted(all_codes): 
                      if ":" in code :
                        all_codes.discard(code[:code.index(":")])
                    self.words[i_subtoken]['PARSEME:MWE'] = ";".join(sorted(all_codes))
            if mustwarn : # warning appears here
                warn("Contraction {} ({}) should not contain MWE annotation {} ".format(c["ID"],c["FORM"],c["PARSEME:MWE"]))

    def iter_mwe_fields_and_normalizedindexes(self, field_name: str, fallback_field_name="FORM"):
        r'''Yield one frozenset[(field_value: str, index: int)] for each MWE in
        this sentence, where the value of `index` is normalized to start at 0.
        '''
        for mweinfo in self.mwe_infos().values():            
            yield mweinfo.field_and_normalizedindex_pairs(field_name, fallback_field_name)

    def iter_mwe_fields_including_span(self, field_name: str, fallback_field_name="FORM"):
        r'''Yield a tuple[str] for each MWE in this sentence.
        If the MWE contains gaps, the words inside those gaps appear in the tuple.
        '''
        for mweinfo in self.mwe_infos().values():            
            yield mweinfo.field_including_span(field_name,fallback_field_name)


class FrozenCounter(collections.Counter):
    r'''Instance of Counter that can be hashed. Should not be modified.'''
    def __hash__(self):
        return hash(frozenset(self.items()))




class MWEInfo(collections.namedtuple('MWEInfo', 'sentence category token_indexes')):
    r"""Represents a single MWE in a sentence.
    CAREFUL: token indexes start at 0 (not at 1, as in the TokenID's).

    Arguments:
    @type sentence: TSVSentence
    @type category: Optional[str]
    @type token_indexes: list[int]
    """
    def n_gaps(self):
        r'''Return the number of gaps inside self.'''
        span_elems = max(self.token_indexes)-min(self.token_indexes)+1        
        assert span_elems >= self.n_tokens(), self        
        return span_elems - self.n_tokens()

    def n_tokens(self):
        r'''Return the number of tokens in self.'''
        return len(self.token_indexes)
    
    def field_and_normalizedindex_pairs(self, field_name: str, fallback_field_name: str):
        r'''Return a frozenset[(field_value: str, index: int)],
        where the value of `index` is normalized to start at 0.
        '''
        min_index = min(self.token_indexes)
        return frozenset((self.sentence.words[i]\
                              .get_fallback(field_name, fallback_field_name), i-min_index)
                         for i in self.token_indexes)

    def field_including_span(self, field_name: str, fallback_field_name: str):
        r'''Return a tuple[str] with all words in this MWE (including words inside its gaps).'''
        first, last = min(self.token_indexes), max(self.token_indexes)
        return tuple(self.sentence.words[i].get_fallback(field_name, fallback_field_name)
                     for i in range(first, last+1))


class TSVToken(collections.UserDict):
    r"""Represents a token in the TSV file.
    You can index this object to get the value of a given field
    (e.g. self["FORM"] or self["PARSEME:MWE"]).

    Extra attributes:
    @type lineno: int
    """
    def __init__(self, lineno, data):
        self.lineno = lineno
        super().__init__(data)
        
    def get_fallback(self, field_name: str, fallback_field_name: str):
        r"""Same as self[field_name], falls back if absent."""
        return self.get(field_name,self.get(fallback_field_name,"_"))
    
    def mwe_codes(self):
        r"""Return a set of MWE codes."""
        mwes = self['PARSEME:MWE']
        return set(mwes.split(';') if mwes != SINGLEWORD else ())

    def mwes_id_categ(self):
        r"""For each MWE code in `self.mwe_codes`, yield an (id, categ) pair.
        @rtype Iterable[(int, Optional[str])]
        """
        for mwe_code in sorted(self.mwe_codes()):
            yield mwe_code_to_id_categ(mwe_code)

    def is_contraction(self):
        r"""Return True iff this token represents a range of tokens.
        (The following tokens in the TSVSentence will contain its elements).
        """
        return "-" in self.get('ID', '')

    def contraction_range(self):
        r"""Return a pair (beg, end) with the
        0-based indexes of the tokens inside this range.
        Should only be called if self.is_contraction() is true.
        """
        assert self.is_contraction()
        a, b = self['ID'].split("-")
        return range(int(a)-1, int(b))

    def __missing__(self, key):
        raise KeyError('''Field {} is underspecified ("_" or missing)'''.format(key))


def mwe_code_to_id_categ(mwe_code):
    r"""mwe_code_to_id_categ(mwe_code) -> (mwe_id, mwe_categ)"""
    split = mwe_code.split(":", 1)
    mwe_id = int(split[0])
    mwe_categ = (split[1] if len(split) > 1 else None)
    return mwe_id, mwe_categ



############################################################


def iter_tsv_sentences(fileobj):
    r"""Yield `TSVSentence` instances for all sentences in the underlying PARSEME TSV file."""
    header = next(fileobj)
    if not 'global.columns' in header:
        exit('ERROR: {}: file is not in the required format: missing global.columns header' \
             .format(os.path.basename(fileobj.name) if len(fileobj.name)>30 else fileobj.name))
    colnames = header.split('=')[-1].split()

    sentence = None
    for lineno, line in enumerate(fileobj, 2):
        global_last_lineno(fileobj.name, lineno)
        if line.startswith("#"):
            pass  # Skip comments
        elif line.strip():
            if not sentence:
                sentence = TSVSentence(fileobj.name, lineno)
            fields = line.strip().split('\t')
            if len(fields) != len(colnames):
                raise Exception('Line has {} columns, but header specifies {}' \
                                .format(len(fields), len(colnames)))
            data = {c: f for (c, f) in zip(colnames, fields) if f != UNDERSP}
            sentence.append(TSVToken(lineno, data))
        else:
            if sentence:
                yield sentence
                sentence = None
    if sentence:
        yield sentence


####################################################################

def write_tsv(sentences, *, file=sys.stdout, fields=CONLLUP_FIELDS):
    r"""Write sentences in TSV format."""
    print("# global.columns =", " ".join(fields), file=file)
    for sentence in sentences:
        for token in sentence.words:
            print(*[token.get(field, "_") for field in fields], sep="\t", file=file)
        print()


####################################################################

last_filename = None
last_lineno = 0

def global_last_lineno(filename, lineno):
    # Update global `last_lineno` var
    global last_filename
    global last_lineno
    last_filename = filename
    last_lineno = lineno


_MAX_WARNINGS = 10
_WARNED = collections.defaultdict(int)

def warn(message, *, warntype="WARNING", position=None, **format_args):
    _WARNED[message] += 1
    if _WARNED[message] <= _MAX_WARNINGS:
        if position is None:
            position = "{}:{}: ".format(last_filename, last_lineno) if last_filename else ""
        msg_list = message.format(**format_args).split("\n")
        if _WARNED[message] == _MAX_WARNINGS:
            msg_list.append("(Skipping following warnings of this type)")

        line_beg, line_end = ('\x1b[31m', '\x1b[m') if COLOR_STDERR else ('', '')
        for i, msg in enumerate(msg_list):
            warn = warntype if i==0 else "."*len(warntype)
            print(line_beg, position, warn, ": ", msg, line_end, sep="", file=sys.stderr)

def excepthook(exctype, value, tb):
    global last_lineno
    global last_filename
    if value and last_lineno:
        last_filename = last_filename or "???"
        err_msg = "===> ERROR when reading {} (line {})" \
                .format(last_filename, last_lineno)
        if COLOR_STDERR:
            err_msg = "\x1b[31m{}\x1b[m".format(err_msg)
        print(err_msg, file=sys.stderr)
    return sys.__excepthook__(exctype, value, tb)


#####################################################################

if __name__ == "__main__":
    sys.excepthook = excepthook
    with open(sys.argv[1]) as f:
        for tsv_sentence in iter_tsv_sentences(f):
            print("TSVSentence:", tsv_sentence)
            print("MWEs:", tsv_sentence.mwe_infos())
