#!/usr/bin/env python2.7
#coding=utf-8
"""
Measures MWE and noun/verb supersense labeling performance for the DiMSUM 2016 shared task
<http://dimsum16.github.io/>. The MWE and supersense evaluation measures
follow Schneider et al., NAACL-HLT 2015 <http://aclweb.org/anthology/N/N15/N15-1177.pdf>:

    P_MWE = #(valid MWE links)/#(predicted MWE links)
    R_MWE = #(valid MWE links)/#(gold MWE links)
    F_MWE = 2*P_MWE*R_MWE / (P_MWE + R_MWE)
    Acc_MWE = #(correct MWE positional tag of {O, B, I, o, b, i})/#(tokens)

    P_supersense = #(correct supersenses)/#(predicted supersenses)
    R_supersense = #(correct supersenses)/#(gold supersenses)
    F_supersense = 2*P_supersense*R_supersense / (P_supersense + R_supersense)
    Acc_supersense = #(correct label: supersense or no-supersense)/#(tokens)

where supersenses are matched on the first token of each expression.
In addition, a combined measure is computed by microaveraging the MWE and supersense
scores: i.e.,

    P_combined = (#(valid MWE links)+#(correct supersenses))/(#(predicted MWE links)+#(predicted supersenses))
    R_combined = (#(valid MWE links)+#(correct supersenses))/(#(gold MWE links)+#(gold supersenses))
    F_combined = 2*P_combined*R_combined / (P_combined + R_combined)
    Acc_combined = #(correct MWE positional tag and label)/#(tokens)

In addition to these 12 scores, this script produces various other statistics, including
confusion matrices for the supersenses. The code was adapted from mweval.py and ssteval.py
in AMALGrAM <https://github.com/nschneid/pysupersensetagger/>.

Usage: ./dimsumeval.py [-p] [-C] test.gold test.pred [test.pred2 ...]

Arguments are files in the 9-column format. Examples have been provided in
the same directory as this script. 2 file arguments corresponds
to evaluating a single system. With >2 file arguments, multiple systems
are compared (color-coding indicates whether scores are higher or lower
than the first system). Confusion matrices and other details are shown
only for the 2-file scenario.

Optional flags:

  -p: Print human-readable gold and predicted analyzed sentences

  -C: Do not colorize the output

TODO: macroaverage by domain (using sentence ID)

@author: Nathan Schneider (nschneid@cs.cmu.edu)
@since: 2015-11-07
"""

from __future__ import print_function, division

import json, os, sys, fileinput, codecs, re
from collections import defaultdict, Counter, namedtuple

from tags2sst import readsents, render



class Ratio(object):
    '''
    Fraction that prints both the ratio and the float value.
    fractions.Fraction reduces e.g. 378/399 to 18/19. We want to avoid this.
    '''
    def __init__(self, numerator, denominator):
        self._n = numerator
        self._d = denominator
    def __float__(self):
        return self._n / self._d if self._d!=0 else float('nan')
    def __str__(self):
        return '{}/{}={:.4f}'.format(self.numeratorS, self.denominatorS, float(self))
    __repr__ = __str__
    def __add__(self, v):
        if v==0:
            return self
        if isinstance(v,Ratio) and self._d==v._d:
            return Ratio(self._n + v._n, self._d)
        return float(self)+float(v)
    def __mul__(self, v):
        return Ratio(self._n * float(v), self._d)
    def __truediv__(self, v):
        return Ratio(self._n / float(v) if float(v)!=0 else float('nan'), self._d)
    __rmul__ = __mul__
    @property
    def numerator(self):
        return self._n
    @property
    def numeratorS(self):
        return ('{:.2f}' if isinstance(self._n, float) else '{}').format(self._n)
    @property
    def denominator(self):
        return self._d
    @property
    def denominatorS(self):
        return ('{:.2f}' if isinstance(self._d, float) else '{}').format(self._d)

def is_tag(t):
    return t in {'B','b','O','o','I','i'}

def f1(prec, rec):
    return 2*prec*rec/(prec+rec) if prec+rec>0 else float('nan')


RE_TAGGING = re.compile(r'^(O|B(o|b[iīĩ]+|[IĪĨ])*[IĪĨ]+)+$'.decode('utf-8'))

def require_valid_mwe_tagging(tagging, kind='tagging'):
    """Verifies the chunking is valid."""

    # check regex
    assert RE_TAGGING.match(''.join(tagging).decode('utf-8')),kind+': '+''.join(tagging)


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



def mweval_sent(sent, ggroups, pgroups, gmwetypes, pmwetypes, stats, indata=None):

    # verify the taggings are valid
    for k,kind in [(1,'gold'),(2,'pred')]:
        tags = zip(*sent)[k]
        require_valid_mwe_tagging(tags, kind=kind)

    if indata:
        gdata, pdata = indata
        stats['Gold_#Groups'] += len(gdata["_"])
        stats['Gold_#GappyGroups'] += sum(1 for grp in gdata["_"] if max(grp)-min(grp)+1!=len(grp))
        if "lemmas" in gdata:
            for grp in gdata["_"]:
                gmwetypes['_'.join(gdata["lemmas"][i-1] for i in grp)] += 1
        stats['Pred_#Groups'] += len(pdata["_"])
        stats['Pred_#GappyGroups'] += sum(1 for grp in pdata["_"] if max(grp)-min(grp)+1!=len(grp))
        for grp in pdata["_"]:
            pmwetypes['_'.join(pdata["lemmas"][i-1] for i in grp)] += 1

    glinks, plinks = [], []
    g_last_BI, p_last_BI = None, None
    g_last_bi, p_last_bi = None, None
    for i,(tkn,goldTag,predTag) in enumerate(sent):

        if goldTag!=predTag:
            stats['incorrect'] += 1
        else:
            stats['correct'] += 1

        if goldTag=='I':
            glinks.append((g_last_BI, i))
            g_last_BI = i
        elif goldTag=='B':
            g_last_BI = i
        elif goldTag=='i':
            glinks.append((g_last_bi, i))
            g_last_bi = i
        elif goldTag=='b':
            g_last_bi = i

        if goldTag in {'O','o'}:
            stats['gold_Oo'] += 1
            if predTag in {'O', 'o'}:
                stats['gold_pred_Oo'] += 1
        else:
            stats['gold_non-Oo'] += 1
            if predTag not in {'O', 'o'}:
                stats['gold_pred_non-Oo'] += 1
                if (goldTag in {'b','i'})==(predTag in {'b','i'}):
                    stats['gold_pred_non-Oo_in-or-out-of-gap_match'] += 1
                if (goldTag in {'B','b'})==(predTag in {'B','b'}):
                    stats['gold_pred_non-Oo_Bb-v-Ii_match'] += 1
                if goldTag in {'I','i'} and predTag in {'I','i'}:
                    stats['gold_pred_Ii'] += 1


        if predTag=='I':
            plinks.append((p_last_BI, i))
            p_last_BI = i
        elif predTag=='B':
            p_last_BI = i
        elif predTag=='i':
            plinks.append((p_last_bi, i))
            p_last_bi = i
        elif predTag=='b':
            p_last_bi = i

    glinks1 = [(a,b) for a,b in glinks]
    plinks1 = [(a,b) for a,b in plinks]
    ggroups1 = [[k-1 for k in g] for g in ggroups]
    assert ggroups1==map(sorted, form_groups(glinks1)),('Possible mismatch between gold MWE tags and parent offsets',ggroups1,glinks1)
    pgroups1 = [[k-1 for k in g] for g in pgroups]
    assert pgroups1==map(sorted, form_groups(plinks1)),('Possible mismatch between predicted MWE tags and parent offsets',pgroups1,plinks1)

    # soft matching (in terms of links)
    stats['PNumer'] += sum(1 for a,b in plinks1 if any(a in grp and b in grp for grp in ggroups1))
    stats['PDenom'] += len(plinks1)
    stats['CrossGapPNumer'] += sum((1 if b-a>1 else 0) for a,b in plinks1 if any(a in grp and b in grp for grp in ggroups1))
    stats['CrossGapPDenom'] += sum((1 if b-a>1 else 0) for a,b in plinks1)
    stats['RNumer'] += sum(1 for a,b in glinks1 if any(a in grp and b in grp for grp in pgroups1))
    stats['RDenom'] += len(glinks1)
    stats['CrossGapRNumer'] += sum((1 if b-a>1 else 0) for a,b in glinks1 if any(a in grp and b in grp for grp in pgroups1))
    stats['CrossGapRDenom'] += sum((1 if b-a>1 else 0) for a,b in glinks1)

    # exact matching (in terms of full groups)
    stats['ENumer'] += sum(1 for grp in pgroups1 if grp in ggroups1)
    stats['EPDenom'] += len(pgroups1)
    stats['ERDenom'] += len(ggroups1)

    for grp in pgroups1:
        gappiness = 'ng' if max(grp)-min(grp)+1==len(grp) else 'g'
        stats['Pred_'+gappiness] += 1



def ssteval_sent(sent, glbls, plbls, sststats, conf):

    def lbl2pos(lbl): return lbl.split('.')[0].lower()  # should be "n" or "v"

    sstpositions = set(glbls.keys()+plbls.keys())

    sststats['Exact Tag']['nGold'] += len(sent)
    sststats['Exact Tag']['tp'] += len(sent) - len(sstpositions)

    for k in sstpositions:
        g = glbls.get(k)
        p = plbls.get(k)
        conf[g,p] += 1

        if g:
            sststats[None]['nGold'] += 1
            sststats[lbl2pos(g)]['nGold'] += 1
        if p:
            sststats[None]['nPred'] += 1
            sststats[lbl2pos(p)]['nPred'] += 1
            if g==p:
                sststats['Exact Tag']['tp'] += 1
                sststats[None]['tp'] += 1
                sststats[lbl2pos(g)]['tp'] += 1

    sststats['Exact Tag']['Acc'] = Ratio(sststats['Exact Tag']['tp'], sststats['Exact Tag']['nGold'])
    for x in sststats:
        if x!='Exact Tag':
            sststats[x]['P'] = Ratio(sststats[x]['tp'], sststats[x]['nPred'])
            sststats[x]['R'] = Ratio(sststats[x]['tp'], sststats[x]['nGold'])
            sststats[x]['F'] = f1(sststats[x]['P'], sststats[x]['R'])

class Colors(object):
    """Terminal color codes. See http://misc.flogisoft.com/bash/tip_colors_and_formatting"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    ORANGE = '\033[93m'
    YELLOW = '\033[33m'
    BLUE = '\033[94m'
    PINK = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BLACK = '\033[30m'
    ENDC = '\033[0m'    # end color
    BLACKBG = '\033[40m'
    WHITEBG = '\033[107m'
    BACKGROUND = BLACKBG
    PLAINTEXT = WHITE

class Styles(object):
    """Terminal style codes."""
    UNDERLINE = '\033[4m'
    NORMAL = '\033[24m'   # normal style: not underlined or bold

SPECTRUM = [Colors.BLUE,Colors.CYAN,Colors.GREEN,Colors.YELLOW,Colors.ORANGE,Colors.RED,Colors.PINK]

def relativeColor(a, b):
    """Compare a value (a) to a baseline/reference value (b), and choose
    a color depending on which is greater."""
    delta = float(a)-float(b)
    if delta>0:
        return Colors.GREEN
    elif delta<0:
        return Colors.ORANGE
    return Colors.PLAINTEXT

def color_render(*args, **kwargs):
    # terminal colors
    WORDS = Colors.YELLOW
    VERBS = Colors.RED
    NOUNS = Colors.BLUE
    MWE = Colors.PLAINTEXT

    s = render(*args, **kwargs)
    c = WORDS+s.replace('_',MWE+'_'+WORDS)+Colors.PLAINTEXT
    c = re.sub(r'(\|v.\w+)', VERBS+r'\1'+WORDS, c)   # verb supersenses
    c = re.sub(r'(\|n.\w+)', NOUNS+r'\1'+WORDS, c)   # noun supersenses

    return c

if __name__=='__main__':
    args = sys.argv[1:]
    printSents = False
    while args and args[0].startswith('-'):
        if args[0]=='-p':   # print sentences to stderr
            printSents = True
        elif args[0]=='-C': # turn off colors
            for c in dir(Colors):
                if not c.startswith('_'):
                    setattr(Colors, c, '')
            for s in dir(Styles):
                if not s.startswith('_'):
                    setattr(Styles, s, '')
            SPECTRUM = ['']
        else:
            assert False,'Unexpected option: '+args[0]
        args = args[1:]

    # set up color defaults
    print(Colors.BACKGROUND + Colors.PLAINTEXT, end='')

    nToks = 0

    goldLblsC = Counter()



    sent = []
    goldFP = args[0]
    predFs = [readsents(fileinput.input(predFP)) for predFP in args[1:]]
    statsCs = [Counter() for predFP in args[1:]]
    sststatsCs = [defaultdict(Counter) for predF in args[1:]]
    gmwetypesCs = [Counter() for predFP in args[1:]]    # these will all have the same contents
    pmwetypesCs = [Counter() for predFP in args[1:]]
    confCs = [Counter() for predFP in args[1:]]    # confusion matrix

    for sentId,gdata in readsents(fileinput.input(goldFP)):
        gtags_mwe = [t.encode('utf-8') for t in gdata["tags"]]
        assert all(len(t)<=1 for t in gtags_mwe)
        glbls = {k-1: v[1].encode('utf-8') for k,v in gdata["labels"].items()}
        goldLblsC.update(glbls.values())
        for predF,stats,gmwetypes,pmwetypes,sststats,conf in zip(predFs,statsCs,gmwetypesCs,pmwetypesCs,sststatsCs,confCs):
            sentId,pdata = next(predF)
            ptags_mwe = [t.encode('utf-8') for t in pdata["tags"]]
            plbls = {k-1: v[1].encode('utf-8') for k,v in pdata["labels"].items()}
            assert all(len(t)<=1 for t in ptags_mwe)
            words, poses = zip(*gdata["words"])
            assert len(words)==len(gtags_mwe)==len(ptags_mwe)
            nToks += len(words)
            stats['nFullTagCorrect'] += sum(1 for k in range(len(words)) if gtags_mwe[k]==ptags_mwe[k] and glbls.get(k)==plbls.get(k))
            if printSents:
                if predFs[0] is predF:
                    print(color_render(words, gdata["_"], [], {k+1: v for k,v in glbls.items()}), file=sys.stderr)
                print(color_render(words, pdata["_"], [], {k+1: v for k,v in plbls.items()}), file=sys.stderr)
            try:
                mweval_sent(zip(words,gtags_mwe,ptags_mwe), gdata["_"], pdata["_"],
                            gmwetypes, pmwetypes, stats, indata=(gdata,pdata))

                ssteval_sent(words, glbls, plbls, sststats, conf)
            except AssertionError as ex:
                print(render(words, gdata["_"], []))
                print(render(words, pdata["_"], []))
                raise ex

    # loaded all files and sentences.
    gmwetypes = gmwetypesCs[0]

    sysprefixes = [('SYS{:0'+str(len(str(len(predFs))))+'}  ').format(i+1) if len(predFs)>1 else '' for i in range(len(predFs))]
    syspad = ' '*len(sysprefixes[0])

    # MWE stats
    print(syspad+'   P   |   R   |   F   |   EP  |   ER  |   EF  |  Acc  |   O   | non-O | ingap | B vs I')
    for stats,conf,pmwetypes,sysprefix in zip(statsCs,confCs,pmwetypesCs,sysprefixes):
        fullAcc = Ratio(stats['nFullTagCorrect'], nToks)

        nTags = stats['correct']+stats['incorrect']
        stats['Acc'] = Ratio(stats['correct'], nTags)
        stats['Tag_R_Oo'] = Ratio(stats['gold_pred_Oo'], stats['gold_Oo'])
        stats['Tag_R_non-Oo'] = Ratio(stats['gold_pred_non-Oo'], stats['gold_non-Oo'])
        stats['Tag_Acc_non-Oo_in-gap'] = Ratio(stats['gold_pred_non-Oo_in-or-out-of-gap_match'], stats['gold_pred_non-Oo'])
        stats['Tag_Acc_non-Oo_B-v-I'] = Ratio(stats['gold_pred_non-Oo_Bb-v-Ii_match'], stats['gold_pred_non-Oo'])
        stats['Tag_Acc_I_strength'] = Ratio(stats['gold_pred_Ii_strength_match'], stats['gold_pred_Ii'])


        stats['P'] = Ratio(stats['PNumer'], stats['PDenom'])
        stats['R'] = Ratio(stats['RNumer'], stats['RDenom'])
        stats['F'] = f1(stats['P'], stats['R'])
        stats['CrossGapP'] = stats['CrossGapPNumer']/stats['CrossGapPDenom'] if stats['CrossGapPDenom']>0 else float('nan')
        stats['CrossGapR'] = stats['CrossGapRNumer']/stats['CrossGapRDenom'] if stats['CrossGapRDenom']>0 else float('nan')
        stats['EP'] = Ratio(stats['ENumer'], stats['EPDenom'])
        stats['ER'] = Ratio(stats['ENumer'], stats['ERDenom'])
        stats['EF'] = f1(stats['EP'], stats['ER'])

        if gmwetypes:
            assert stats['Gold_#Groups']==sum(gmwetypes.values())
            stats['Gold_#Types'] = len(gmwetypes)
        assert stats['Pred_#Groups']==sum(pmwetypes.values())
        stats['Pred_#Types'] = len(pmwetypes)

        if len(predFs)==1:
            print('mwestats = ', dict(stats), ';', sep='')
            print()
            print('sststats = ', dict(sststats), ';', sep='')
            print()
            print('conf = ', dict(conf), ';', sep='')
            print()

        parts = [(' {1}{0:.2%}'.format(float(stats[x]), relativeColor(stats[x],statsCs[0][x]))+Colors.PLAINTEXT,
                  '{:>7}'.format('' if x.endswith('F') or isinstance(stats[x],(float,int)) else stats[x].numeratorS),
                  '{:>7}'.format('' if x.endswith('F') or isinstance(stats[x],(float,int)) else stats[x].denominatorS)) for x in ('P', 'R', 'F', 'EP', 'ER', 'EF', 'Acc',
                  'Tag_R_Oo', 'Tag_R_non-Oo',
                  'Tag_Acc_non-Oo_in-gap', 'Tag_Acc_non-Oo_B-v-I')]
        for j,pp in enumerate(zip(*parts)):
            print((sysprefix if j==0 else syspad)+' '.join(pp))
    print()

    #print(pmwetypes)

    # Supersense stats
    if len(predFs)==1:
        # supersense confusion matrices
        colrs = {'n.': Colors.RED, 'v.': Colors.BLUE}
        fmts = {'n.': str.upper, 'v.': str.lower}
        for d,d2 in (('n.','v.'),('v.','n.')):
            matrix = [['{: >15}'.format('----')+' {:5}'.format(goldLblsC[None] or '')]]
            header = ['           {}GOLD{}      '.format(Styles.UNDERLINE, Styles.NORMAL),' ----']
            lbls = [None]
            for lbl,n in goldLblsC.most_common():
                if lbl.startswith(d):
                    lbls.append(lbl)
                    matrix.append([colrs[d]+'{: >15}'.format(lbl)+Colors.PLAINTEXT+' {:5}'.format(n)])
                    header.append(' '+colrs[d]+fmts[d](lbl[2:])[:4]+Colors.PLAINTEXT)
            # cross-POS confusions
            gconfsC = Counter([p for (g,p),n in conf.most_common() if g and p and g.startswith(d) for i in range(n)])
            for lbl,n in sorted(gconfsC.most_common(), key=lambda (l,lN): not l.startswith(d)):
                if lbl not in lbls:
                    lbls.append(lbl)
                    #matrix.append([colrs[d2]+'{: >15}'.format(lbl)+Colors.PLAINTEXT+' {:5}'.format(n)])
                    header.append(' '+colrs[lbl[:2]]+fmts[lbl[:2]](lbl[2:])[:4]+Colors.PLAINTEXT)
                    # since this label is for the other part of speech, show as a column (predicted) but not a row (gold)

            header.append(' <-- PRED')

            # matrix content
            if not conf:
                print(Colors.RED+'No gold or predicted supersenses found: check that the input is in the right format. Exiting.'+Colors.RED+Colors.ENDC)
                sys.exit(1)
            nondiag_max = [n for (g,p),n in conf.most_common() if (g is None or g.startswith(d)) and g!=p][0]

            for i,g in enumerate(lbls):
                if i>=len(matrix): continue
                for j,p in enumerate(lbls):
                    while len(matrix[i])<=j+1:
                        matrix[i].append('')
                    v = conf[g,p]
                    #if v>0 or i==j:
                    #    print(v, g,p, int((v-1)/nondiag_max*len(SPECTRUM)), nondiag_max)
                    colr = SPECTRUM[int((v-1)/nondiag_max*len(SPECTRUM))] if v>0 and i!=j else Colors.PLAINTEXT
                    matrix[i][j+1] = colr+' {:4}'.format(conf[g,p] or '')+Colors.PLAINTEXT

            print(''.join(header))
            for ln in matrix:
                print(''.join(ln))
            print()

    # supersense scores
    print(syspad+'  Acc  |   P   |   R   |   F   || R: NSST | VSST ')
    for sststats,sysprefix in zip(sststatsCs,sysprefixes):
        parts = [(' {1}{0:.2%}'.format(float(sststats['Exact Tag']['Acc']), relativeColor(sststats['Exact Tag']['Acc'],sststatsCs[0]['Exact Tag']['Acc']))+Colors.PLAINTEXT,
                  '{:>7}'.format(sststats['Exact Tag']['Acc'].numeratorS),
                  '{:>7}'.format(sststats['Exact Tag']['Acc'].denominatorS))]
        parts += [(' {1}{0:.2%}'.format(float(sststats[None][x]), relativeColor(sststats[None][x],sststatsCs[0][None][x]))+Colors.PLAINTEXT,
                   '{:>7}'.format(sststats[None][x].numeratorS),
                   '{:>7}'.format(sststats[None][x].denominatorS)) for x in ('P', 'R')]
        parts += [(' {1}{0:.2%}  '.format(float(sststats[None]['F']), relativeColor(sststats[None]['F'],sststatsCs[0][None]['F']))+Colors.PLAINTEXT,
                   '         ',
                   '         ')]
        parts += [(' {1}{0:.2%}'.format(float(sststats[y]['R']), relativeColor(sststats[y]['R'],sststatsCs[0][y]['R']))+Colors.PLAINTEXT,
                   '{:>7}'.format(sststats[y]['R'].numeratorS),
                   '{:>7}'.format(sststats[y]['R'].denominatorS)) for y in ('n', 'v')]
        for j,pp in enumerate(zip(*parts)):
            print((sysprefix if j==0 else syspad)+' '.join(pp))
    print()

    # combined acc, P, R, F
    print(syspad+'  Acc  |   P   |   R   |   F   ')
    cstatsBL = None
    for sststats,sysprefix in zip(sststatsCs,sysprefixes):
        cstats = Counter()
        cstats['Acc'] = fullAcc
        cstats['P'] = Ratio(stats['P'].numerator + sststats[None]['P'].numerator,
                            stats['P'].denominator + sststats[None]['P'].denominator)
        cstats['R'] = Ratio(stats['R'].numerator + sststats[None]['R'].numerator,
                            stats['R'].denominator + sststats[None]['R'].denominator)
        cstats['F'] = f1(cstats['P'], cstats['R'])
        if cstatsBL is None:
            cstatsBL = cstats

        parts = [(' {1}{0:.2%}'.format(float(cstats[x]), relativeColor(cstats[x],cstatsBL[x]))+Colors.PLAINTEXT,
                  '{:>7}'.format('' if x.endswith('F') or isinstance(cstats[x],(float,int)) else cstats[x].numeratorS),
                  '{:>7}'.format('' if x.endswith('F') or isinstance(cstats[x],(float,int)) else cstats[x].denominatorS)) for x in ('Acc', 'P', 'R', 'F')]
        for j,pp in enumerate(zip(*parts)):
            print((sysprefix if j==0 else syspad)+' '.join(pp))

    if len(predFs)==1:
        print()
        print('SUMMARY SCORES')
        print('==============')
        print(re.sub(r'=([^=]+)$', '='+Colors.YELLOW+r'\1'+Colors.PLAINTEXT, 'MWEs: P={stats[P]} R={stats[R]} F={f:.2%}'.format(stats=stats, f=float(stats['F']))))
        print(re.sub(r'=([^=]+)$', '='+Colors.PINK+r'\1'+Colors.PLAINTEXT, 'Supersenses: P={stats[P]} R={stats[R]} F={f:.2%}'.format(stats=sststats[None], f=float(sststats[None]['F']))))
        print(re.sub(r'=([^=]+)$', '='+Colors.GREEN+r'\1'+Colors.PLAINTEXT, 'Combined: Acc={stats[Acc]} P={stats[P]} R={stats[R]} F={f:.2%}'.format(stats=cstats, f=float(cstats['F']))))

    # restore the terminal's default colors
    print(Colors.ENDC, end='')
