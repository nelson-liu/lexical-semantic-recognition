#!/usr/bin/env python2.7
#coding=utf-8
'''
Converts a file in the .sst format to the official format of the DiMSUM 2016 shared task
<http://dimsum16.github.io/> (one word per line using the OoBbIi encoding for MWEs
and a separate column for supersense labels). If the sentence cannot be represented in
this encoding, some groupings will be removed and a message written to stderr.
tags2sst.py goes in the opposite direction.

Input format (3 columns; additional fields may be present in the JSON object but will be ignored):

sentID   annotated_sentence   {"words": [[word1,pos1],...], "labels": {"offset1": [word1,label1], "offset2": [word2,label2]}, "_": [[offset1,offset2,offset3]}

Output is in the tab-separated format:

offset   word   [lemma]   POS   tag   parent   (blank)   label   sentId

@author: Nathan Schneider (nschneid@cs.cmu.edu)
@since: 2015-10-11
'''
from __future__ import print_function, division
import os, sys, re, fileinput, codecs, json

def convert(inF, outF=sys.stdout):

    for ln in inF:
        sentId, anno, data = ln.rstrip().split('\t')
        data = json.loads(data)
        parents = {}
        gapstrength = {}    # offset -> '_' if the offset lies with a gap

        # process strong groups
        for grp in data["_"]:
            g = sorted(grp)
            skip = False
            for i,j in zip(g[:-1],g[1:]):
                if j>i+1:
                    if i in gapstrength:    # gap within a gap
                        print('Simplifying: removing gappy group that is wholly contained within another gap:', g, anno, file=sys.stderr)
                        skip = True
                        break
            if skip: continue

            for i,j in zip(g[:-1],g[1:]):
                assert j not in parents
                parents[j] = i, '_'
                if j>i+1:
                    for h in range(i+1,j):
                        gapstrength[h] = '_'

        allparents = set(zip(*parents.values())[0]) if parents else set()

        labels = {int(k): l for k,(w,l) in data["labels"].items()}
        for i,(w,pos) in enumerate(data["words"]):
            parent, strength = parents.get(i+1,(0,''))
            label = labels.get(i+1,'')
            labelFlag = '-'+label if label else ''
            amInGap = (i+1 in gapstrength)

            if parent==0:
                assert strength==''
                if i+1 in allparents:
                    tag = ('b' if amInGap else 'B')
                else:
                    tag = ('o' if amInGap else 'O')
            else:
                assert strength=='_'    # No weak MWEs in this dataset.
                tag = 'i' if amInGap else 'I'

            lemma = data["lemmas"][i]

            print(i+1, w.encode('utf-8'), lemma.encode('utf-8'), pos, tag.encode('utf-8'),
                  parent, 
                  '', # Don't print strength.
                  label, sentId, sep='\t', file=outF)
        print()

if __name__=='__main__':
    convert(fileinput.input())
