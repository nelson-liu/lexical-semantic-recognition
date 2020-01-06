#!/usr/bin/env bash

set -e

for modelpath in models/streusle_*; do
    modelname=$(basename ${modelpath})
    echo "Evaluating ${modelname}"
    allennlp predict models/${modelname}/model.tar.gz data/dimsum16/dimsum16.test.blind.jsonl \
        --silent \
        --output-file models/${modelname}/${modelname}_dimsum16_test_predictions.jsonl \
        --silent \
        --include-package streusle_tagger \
        --predictor streusle-tagger \
        --cuda-device 0 \
        --batch-size 64

    ./scripts/evaluate_dimsum_predictions.sh \
        "models/${modelname}/${modelname}_dimsum16_test_predictions.jsonl" \
        ./data/dimsum16/dimsum16.test
done
