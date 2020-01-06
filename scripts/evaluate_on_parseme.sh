#!/usr/bin/env bash

set -e

for modelpath in models/streusle_*; do
    modelname=$(basename ${modelpath})
    echo "Evaluating ${modelname}"
    allennlp predict models/${modelname}/model.tar.gz data/parseme_en/test.blind.jsonl \
        --silent \
        --output-file models/${modelname}/${modelname}_parseme_en_test_predictions.jsonl \
        --silent \
        --include-package streusle_tagger \
        --predictor streusle-tagger \
        --cuda-device 0 \
        --batch-size 64

    ./scripts/evaluate_parseme_predictions.sh \
        "models/${modelname}/${modelname}_parseme_en_test_predictions.jsonl" \
        ./data/parseme_en/test.cupt
done
