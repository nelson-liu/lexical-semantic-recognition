#!/usr/bin/env python2.7
#coding=utf-8
"""
Implements a function, readsents(), that reads a file in the 9-column format of the DiMSUM 2016
shared task <http://dimsum16.github.io/> (one word per line using the OoBbIi encoding
for MWEs and a separate column for supersense labels).
Also implements render(), which displays a human-readable version of an analyzed sentence.

Input is in the tab-separated format:

offset   word   lemma   POS   MWEtag   parent   (blank)   label   sentId

If run from the command line, produces a .sst file with grouped MWE offsets, one sentence
per line, generating a human-readable annotation of the segmentation
and including MWE groups, labels, lemmas (if available), and tags in a JSON object.
sst2tags.py goes in the opposite direction.

Output format (3 columns):

sentID   annotated_sentence   {"words": [[word1,pos1],...], "labels": {"offset1": [word1,label1], "offset2": [word2,label2]}, "lemmas": [lemma1,lemma2,...], "tags": [tag1,tag2,...], "_": [[offset1,offset2],...], "~": []}

With the -l flag, show supersense labels in annotated_sentence.
Otherwise, annotated_sentence will only contain the segmentation.

The code was adapted from tags2sst.py in AMALGrAM <https://github.com/nschneid/pysupersensetagger/>.
The input format for the task is slightly different from AMALGrAM's
(the 5th column contains only the MWE tag, and there is no strength distinction).

@author: Nathan Schneider (nschneid@cs.cmu.edu)
@since: 2015-10-11
"""

from __future__ import print_function, division
import sys, fileinput, json, StringIO
from __builtin__ import True

def render(ww, sgroups, wgroups, labels={}):
    '''
    Converts the given lexical annotation to a UTF-8 string
    with _ and ~ as weak and strong joiners, respectively.
    Assumes this can be done straightforwardly (no nested gaps,
    no weak expressions involving words both inside and outside
    of a strong gap, no weak expression that contains only
    part of a strong expression, etc.).
    Also does not specially escape of tokens containing _ or ~.

    Note that indices are 1-based.

    >>> ww = ['a','b','c','d','e','f']
    >>> render(ww, [[2,3],[5,6]], [[1,2,3,5,6]])
    'a~b_c~ d ~e_f'
    >>> render(ww, [], [], {3: 'C', 6: 'FFF'})
    'a b c|C d e f|FFF'
    >>> render(ww, [[2,3],[5,6]], [], {2: 'BC', 5: 'EF'})
    'a b_c|BC d e_f|EF'
    >>> render(ww, [[1,2,6],[3,4,5]], [], {1: 'ABF'})
    'a_b_ c_d_e _f|ABF'
    >>> render(ww, [[1,2,6],[3,4,5]], [], {1: 'ABF', 3: 'CDE'})
    'a_b_ c_d_e|CDE _f|ABF'
    >>> render(ww, [], [[3,4,5]], {4: 'D', 5: 'E', 6: 'F'})
    'a b c~d|D~e|E f|F'
    >>> render(ww, [], [[3,5]])
    'a b c~ d ~e f'
    >>> render(ww, [[2,3],[5,6]], [[2,3,4]], {4: 'D'})
    'a b_c~d|D e_f'
    >>> render(ww, [[2,3],[5,6]], [[1,2,3,5,6]])
    'a~b_c~ d ~e_f'
    >>> render(ww, [[2,3],[5,6]], [[1,2,3,4,5,6]], {1: 'A', 2: 'BC', 4: 'D', 5: 'EF'})
    'a|A~b_c|BC~d|D~e_f|EF'
    >>> render(ww, [[2,4],[5,6]], [[2,4,5,6]], {2: 'BD', 3: 'C'})
    'a b_ c|C _d|BD~e_f'
    '''
    singletonlabels = dict(labels)  # will be winnowed down to the labels not covered by a strong MWE
    before = [None]*len(ww)   # None by default; remaining None's will be converted to empty strings
    labelafter = ['']*len(ww)
    after = [None]*len(ww)   # None by default; remaining None's will be converted to spaces
    for group in sgroups:
        g = sorted(group)
        for i,j in zip(g[:-1],g[1:]):
            if j==i+1:
                after[i-1] = ''
                before[j-1] = '_'
            else:
                after[i-1] = '_'
                before[i] = ' '
                before[j-1] = '_'
                after[j-2] = ' '
        if g[0] in labels:
            labelafter[g[-1]-1] = '|'+labels[g[0]]
            del singletonlabels[g[0]]
    for i,lbl in singletonlabels.items():
        assert i-1 not in labelafter
        labelafter[i-1] = '|'+lbl
    for group in wgroups:
        g = sorted(group)
        for i,j in zip(g[:-1],g[1:]):
            if j==i+1:
                if after[i-1] is None and before[j-1] is None:
                    after[i-1] = ''
                    before[j-1] = '~'
            else:
                if after[i-1] is None and before[i] is None:
                    after[i-1] = '~'
                    before[i] = ' '
                if after[j-2] is None and before[j-1] is None:
                    before[j-1] = '~'
                    after[j-2] = ' '

    after = ['' if x is None else x for x in after]
    before = [' ' if x is None else x for x in before]
    return u''.join(sum(zip(before,ww,labelafter,after), ())).strip().encode('utf-8')

def process_sentence(words, lemmas, tags, labels, parents, sentId=None):
    # form groups
    sgroups = []
    wgroups = []
    i2sgroup = {}
    i2wgroup = {}
    for offset,(parent,strength) in sorted(parents.items()):
        if strength in {'_',''}:
            if parent not in i2sgroup:
                i2sgroup[parent] = len(sgroups)
                sgroups.append([parent])
            i2sgroup[offset] = i2sgroup[parent]
            sgroups[i2sgroup[parent]].append(offset)
    for offset,(parent,strength) in sorted(parents.items()):
        if strength=='~':   # includes transitive closure over all member strong groups
            if parent not in i2wgroup:
                i2wgroup[parent] = len(wgroups)
                wgroups.append([])
            i2wgroup[offset] = i2wgroup[parent]
            g = wgroups[i2wgroup[offset]]

            if parent in i2sgroup: # include strong group of parent
                for o in sgroups[i2sgroup[parent]]:
                    if o not in g:  # avoid redundancy if weak group has 3 parts
                        g.append(o)
            elif parent not in g:
                g.append(parent)

            if offset in i2sgroup:  # include strong group of child
                for o in sgroups[i2sgroup[offset]]:
                    i2wgroup[o] = i2wgroup[offset]  # in case the last word in a strong expression precedes part of a weak expression
                    g.append(o)
            else:
                g.append(offset)

    # sanity check: number of tokens belonging to some MWE
    assert len(set(sum(sgroups+wgroups,[])))==sum(1 for t in tags if t[0].upper()!='O'),(sentId,tags,sgroups,wgroups)

    # sanity check: no token shared by multiple strong or multiple weak groups
    assert len(set(sum(sgroups,[])))==len(sum(sgroups,[])),(sgroups,tags,sentId)
    assert len(set(sum(wgroups,[])))==len(sum(wgroups,[])),(wgroups,tags,sentId)

    for k,lbl in enumerate(labels):
        if lbl:
            assert not any(k+1 in g and k+1!=g[0] for g in sgroups), 'Label for a strong group must only be present for its first token: '+lbl+' at '+str(k+1)


    data = {"words": words, "tags": tags, "_": sgroups, "~": wgroups,
            "labels": {k+1: [words[k][0],lbl] for k,lbl in enumerate(labels) if lbl}}
    if any(lemmas):
        data["lemmas"] = lemmas

    return data


def readsent(inF):
    words = []
    lemmas = []
    tags = []
    labels = []
    parents = {}

    for ln in inF:
        if not ln.strip():
            if not words: continue
            break

        assert ln.endswith('\n')
        parts = ln[:-1].decode('utf-8').split('\t')
        assert 8<=len(parts)<=9, parts
        if len(parts)==9:
            offset, word, lemma, POS, tag, parent, strength, label, sentId = parts
        else:
            sentId = ''
            offset, word, lemma, POS, tag, parent, strength, label = parts
        words.append((word, POS))
        lemmas.append(lemma)
        tags.append(tag)
        labels.append(label)
        assert parent
        if int(parent)!=0:
            parents[int(offset)] = (int(parent), strength)

    if not words: raise StopIteration()

    data = process_sentence(words, lemmas, tags, labels, parents, sentId=sentId)
    return sentId,data

def readsents(inF):
    while True:
        try:
            sentId,data = readsent(inF)
            yield sentId,data
        except StopIteration:
            break

def convert(inF, outF=sys.stdout, labelsInRenderedAnno=False):
    for sentId,data in readsents(inF):
        print(sentId,
              render(zip(*data["words"])[0], data["_"], data["~"],
              ({int(k): v[1] for k,v in data["labels"].items()} if labelsInRenderedAnno else {})),
              json.dumps(data), sep='\t', file=outF)


def test():
    """
    >>> test()
    But Dumbledore|n.person says|v.communication he does n't care|v.emotion what they \
do|v.change as_long_as they do|v.change n't take_ him _off|v.change \
the Chocolate_Frog|n.food cards|n.artifact .
    ***
    Would you care_for|v.emotion a lemon_drop|n.food ?
    ***
    Harry|n.person had_ a_lot of _trouble|v.cognition keeping|v.stative his \
mind|n.cognition on his lessons|n.cognition
    ***
    """

    t1 = '''
1 But - CC O 0
2 Dumbledore - NNP O 0  n.person
3 says - VBZ O 0  v.communication
4 he - PRP O 0
5 does - VBZ O 0
6 n't - RB O 0
7 care - VB O 0  v.emotion
8 what - WP O 0
9 they - PRP O 0
10 do - VBP O 0  v.change
11 as - IN B 0
12 long - JJ I 11
13 as - IN I 12
14 they - PRP O 0
15 do - VBP O 0  v.change
16 n't - RB O 0
17 take - VB B 0  v.change
18 him - PRP o 0
19 off - RP I 17
20 the - DT O 0
21 Chocolate - NNP B 0  n.food
22 Frog - NNP I 21
23 cards - NNS O 0  n.artifact
24 . - . O 0

1 Would - MD O 0   sent2
2 you - PRP O 0   sent2
3 care - VB B 0  v.emotion sent2
4 for - IN I 3   ent2
5 a - DT O 0   sent2
6 lemon - NN B 0  n.food sent2
7 drop - NN I 6   sent2
8 ? - . O 0   sent2


1 Harry - NNP O 0  n.person
2 had - VBD B 0  v.cognition
3 a - DT b 0
4 lot - NN i 3
5 of - IN o 0
6 trouble - NN I 2
7 keeping - VBG O 0  v.stative
8 his - PRP$ O 0
9 mind - NN O 0  n.cognition
10 on - IN O 0
11 his - PRP$ O 0
12 lessons - NNS O 0  n.cognition
'''.lstrip().replace(' ','\t')

    for data in readsents(StringIO.StringIO(t1)):
        print(render([w for w,pos in data["words"]], data["_"], data["~"],
                     {int(k): v[1] for k,v in data["labels"].items()}))
        print('***')

if __name__=='__main__':
    #import doctest
    #doctest.testmod()

    args = sys.argv[1:]
    if args and args[0]=='-l':
        labelsInRenderedAnno = True
        args = args[1:]
    else:
        labelsInRenderedAnno = False
    convert(fileinput.input(args), labelsInRenderedAnno=labelsInRenderedAnno)
