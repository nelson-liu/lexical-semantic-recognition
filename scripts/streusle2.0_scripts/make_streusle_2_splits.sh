#!/usr/bin/env bash

python scripts/streusle2.0_scripts/make_streusle2_splits.py \
    --data-path data/streusle2.0/streusle.sst \
    --train-ids-path data/streusle2.0/train.sentids \
    --test-ids-path data/streusle2.0/test.sentids \
    --train-output-path data/streusle2.0/streusle.sst.train \
    --test-output-path data/streusle2.0/streusle.sst.test
