#!/usr/bin/env bash

python scripts/streusle2.0_scripts/make_streusle2_splits.py \
    --data-path data/streusle2.0/streusle.tags.sst \
    --train-ids-path data/streusle2.0/train.sentids \
    --test-ids-path data/streusle2.0/test.sentids \
    --num-dev-sentences 500 \
    --train-output-path data/streusle2.0/streusle.tags.sst.train \
    --test-output-path data/streusle2.0/streusle.tags.sst.test \
    --dev-output-path data/streusle2.0/streusle.tags.sst.dev \
