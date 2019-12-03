#!/usr/bin/env python2.7
#coding=utf-8
'''
Converts tagged sentences into a simpler form without gaps and/or weak links.
If the mode is gaps+weak, 4 versions of each sentence are produced.
Otherwise, 2 vesions are produced, corresponding to liberal and conservative
conversion rules.

Args: gaps|weak|gaps+weak TAGGED_FILE

@author: Nathan Schneider (nschneid@cs.cmu.edu)
@since: 2013-06-30
'''

from __future__ import print_function, division

import json, os, sys, fileinput, codecs, re
from collections import Counter

def is_tag(t):
    return t in {'B','b','O','o','I','I_','I~','i','i_','i~'}

def f1(prec, rec):
    return 2*prec*rec/(prec+rec) if prec+rec>0 else float('nan')

RE_TAGGING = re.compile(r'^(O|B(o|b[ii_i~]+|[II_I~])*[II_I~]+)+$')

def require_valid_tagging(tagging, simplify_gaps, simplify_weak):
    assert re.match(r'^(O|B(o|b[ii_i~]+|[II_I~])*[II_I~]+)+$', tagging)
    if simplify_gaps:
        assert re.match(r'^(O|B[II_I~]+)+$', tagging)
    if simplify_weak:
        assert re.match(r'^(O|B(o|bi+|I)*I+)+$', tagging)

IN_GAP_TAGS = {'o','b','i','i_','i~'}
I_TILDE = 'I~'
i_TILDE = 'i~'
I_BAR = 'I_'
i_BAR = 'i_'

def simplify(tags, simplification=None and 'gaps' and 'weak' and 'gaps+weak',
    policy='all' or 'best'):
    '''
    For each possible conversion of the sentence under the given simplification scheme,
    modifies the gold tags and yields the instance weight for the simplified version
    (such that all weights are equal and sum to 1). Then restores the original
    gold tags.
    '''
    assert simplification in {'gaps', 'weak', 'gaps+weak'}
    assert policy in {'all', 'best'}
    BEST_POLICY_RESULT = {'weak': 1, # the high-recall (liberal) policy: convert weak to strong
                          'gaps': 0, # the high-precision (conservative) policy: remove cross-gap links
                          'gaps+weak': 1}   # combination of the above

    gold_tags = set(tags)
    assert gold_tags<=set('OoBbIi') | {'I_','I~','i_','i~'}
    simplify_gaps = simplification in {'gaps','gaps+weak'} and not gold_tags<={'O','B','I','I_','I~'}
    simplify_weak = simplification in {'weak','gaps+weak'} and not gold_tags<=set('OoBbIi')

    results = []

    if simplify_gaps or simplify_weak:
        #print(' '.join(tokens)+'\n'+''.join(tags), file=sys.stderr)

        tt = list(tags) # output tags
        if simplify_gaps:
            assert 'o' in gold_tags or 'b' in gold_tags
            # conservative: remove gaps from gappy expressions
            for i,orig in enumerate(tags):
                if tt[i] in IN_GAP_TAGS:
                    if tt[i-1] not in IN_GAP_TAGS:
                        if tt[i-1]=='B':
                            tt[i-1] = 'O'
                    #tt[i] = 'O'
                    if tt[i+1] not in IN_GAP_TAGS:
                        # a strong or weak I tag
                        tt[i+1] = 'B' if i+2<len(tags) else 'O'
                elif tt[i] in {'O','B'} and i>0 and tt[i-1]=='B':
                    # we introduced a B after a gap which is actually a singleton, so remove it
                    tt[i-1] = 'O'

            result = [t.upper().replace(i_TILDE, I_TILDE).replace(i_BAR, I_BAR) for t in tt]
            #if 'weak' not in simplification: print(''.join(result), file=sys.stderr)
            require_valid_tagging(''.join(result), simplify_gaps, False)
            results.append(result)

            tt = list(tags)
            # liberal: link across gaps (weakly if possible, preserving in-gap strong MWEs)
            for i,orig in enumerate(tags):
                if tt[i] in IN_GAP_TAGS:
                    if simplify_weak:
                        tt[i] = 'I'
                    elif tt[i]=='b':
                        if tt[i-1]=='B':
                            tt[i] = I_TILDE
                    else:
                        if tt[i]==i_TILDE and tags[i-1]=='b':
                            # weak link within a gap: merge it with the cross-gap weak MWE
                            tt[i-1] = I_TILDE
                        if tt[i]!=i_BAR:
                            tt[i] = I_TILDE
                elif not simplify_weak and tt[i]==I_BAR and tags[i-1] in IN_GAP_TAGS:
                    # post-gap continuation should be weak
                    tt[i] = I_TILDE


            result = [t.upper().replace(i_TILDE, I_TILDE).replace(i_BAR, I_BAR) for t in tt]
            #if 'weak' not in simplification: print(''.join(result), file=sys.stderr)
            require_valid_tagging(''.join(result), simplify_gaps, False)
            results.append(result)
        else:
            results.append(list(tags))
            if 'gaps' in simplification:
                results.append(list(tags))

        assert len(results)==(2 if 'gaps' in simplification else 1),results

        if simplify_weak:
            partial_results = results
            results = []
            for partial_result in partial_results:
                # conservative: remove weak links
                tt = list(partial_result)

                # - convert weak I's to B's, and strong I's to plain I's
                tt[:] = [{i_TILDE: 'b', I_TILDE: 'B', i_BAR: 'i', I_BAR: 'I'}.get(t,t) for t in tt]
                # - remove trans-gap weak links
                for i,t in enumerate(tt):
                    if t=='B' and i>0 and tt[i-1] in {'o','i'}:	# B after gap. the trans-gap link was weak, so everything inside the gap becomes no longer gappy
                        j = i-1
                        while tt[j].islower():
                            tt[j] = tt[j].upper()
                            j -= 1
                        if tt[j]=='B':
                            tt[j] = 'O'
                # - remove singleton B's (B must be followed by I or a gap)
                for i,t in enumerate(tt):
                    if t=='B':
                        if i+1==len(tt):    # B at end of sequence
                            tt[i] = 'O'
                        elif i>0 and tt[i-1]=='b':
                            assert False
                        elif tt[i+1] in {'O', 'B'}: # singleton B
                            tt[i] = 'O'
                    elif t=='b':
                        if tt[i+1]!='i':
                            tt[i] = 'o'
                # TODO: weak trans-gap link
                #print(''.join(tt), file=sys.stderr)
                require_valid_tagging(''.join(tt), simplify_gaps, simplify_weak)
                results.append(tt)

                # liberal: convert weak links to strong links
                tt = list(partial_result)
                for i,t in enumerate(tt):
                    if t in {i_TILDE, i_BAR}:
                        tt[i] = 'i'
                    elif t in {I_TILDE, I_BAR}:
                        tt[i] = 'I'

                #print(''.join(tt), file=sys.stderr)
                require_valid_tagging(''.join(tt), simplify_gaps, simplify_weak)
                results.append(tt)
        elif 'weak' in simplification:
            results.append(list(results[0]))
            results.append(list(results[1]))

        assert len(results)==(4 if simplification=='gaps+weak' else 2),(simplify_gaps,simplify_weak,results)

    else:       # nothing to do for this sentence
        for x in range(4 if simplification=='gaps+weak' else 2):
            results.append(list(tags))

    if policy=='best':
        results = [results[BEST_POLICY_RESULT[simplification]]]
    return results

if __name__=='__main__':
    mode = sys.argv[1]
    assert mode in {'gaps','weak','gaps+weak',
                    'gaps_best','weak_best','gaps+weak_best'}

    simplification = mode.split('_')[0]
    policy = 'best' if mode.endswith('_best') else 'all'

    sent = []
    with open(sys.argv[2]) as f:
        for prediction in f:
            prediction_json = json.loads(prediction)
            tags = [x.split("-")[0] for x in prediction_json["tags"]]
            conservative_simplified_tags = simplify(tags, simplification=simplification, policy=policy)[0]
            print(f"Original tags: {tags}")
            print(f"Simplified (conservative) tags: {conservative_simplified_tags}")
