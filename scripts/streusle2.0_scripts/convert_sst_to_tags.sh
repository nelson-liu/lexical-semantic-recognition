#!/usr/bin/env bash

if [[ "$OSTYPE" == "darwin"* ]]; then
    SEDCOMMAND=gsed
else
    SEDCOMMAND=sed
fi

scripts/streusle2.0_scripts/sst2tags.py -l ./data/streusle2.0/streusle.tags.sst.train | ${SEDCOMMAND} -r $'s/-`j?\t/\t/g' > ./data/streusle2.0/streusle.tags.train
scripts/streusle2.0_scripts/sst2tags.py -l ./data/streusle2.0/streusle.tags.sst.test | ${SEDCOMMAND} -r $'s/-`j?\t/\t/g' > ./data/streusle2.0/streusle.tags.test
scripts/streusle2.0_scripts/sst2tags.py -l ./data/streusle2.0/streusle.tags.sst.dev | ${SEDCOMMAND} -r $'s/-`j?\t/\t/g' > ./data/streusle2.0/streusle.tags.dev
cut -f5 ./data/streusle2.0/streusle.tags.train | sort | uniq | ${SEDCOMMAND} '/^\s*$/d' > ./data/streusle2.0/streusle.tags.train.tagset
