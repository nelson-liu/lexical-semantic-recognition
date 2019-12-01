#!/usr/bin/env bash

set -e

for modelpath in models/streusle*; do
    modelname=$(basename ${modelpath})
    echo "Evaluating ${modelname}"
    allennlp predict models/${modelname}/model.tar.gz data/streusle/streusle.ud_dev.json \
        --silent \
        --output-file models/${modelname}/${modelname}_dev_predictions.jsonl \
        --silent \
        --include-package streusle_tagger \
        --use-dataset-reader \
        --predictor streusle-tagger \
        --batch-size 64
done
