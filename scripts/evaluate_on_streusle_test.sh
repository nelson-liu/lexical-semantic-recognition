#!/usr/bin/env bash

set -e

for modelpath in models/streusle_*; do
    modelname=$(basename ${modelpath})
    echo "Evaluating ${modelname}"
    allennlp predict models/${modelname}/model.tar.gz data/streusle/streusle.ud_test.json \
        --silent \
        --output-file models/${modelname}/${modelname}_test_predictions.jsonl \
        --silent \
        --include-package streusle_tagger \
        --use-dataset-reader \
        --predictor streusle-tagger \
        --cuda-device 0 \
        --batch-size 64

    ./scripts/streusle_eval/streuseval.sh \
        data/streusle/streusle.ud_test.conllulex \
        "models/${modelname}/${modelname}_test_predictions.jsonl"

done
