#!/usr/bin/env bash

set -e

for modelpath in models/streusle2_*; do
    modelname=$(basename ${modelpath})
    echo "Evaluating ${modelname}"
    allennlp predict models/${modelname}/model.tar.gz data/streusle2.0/streusle.tags.dev \
        --silent \
        --output-file models/${modelname}/${modelname}_dev_predictions.jsonl \
        --silent \
        --include-package streusle_tagger \
        --use-dataset-reader \
        --predictor streusle-tagger \
        --cuda-device 0 \
        --batch-size 64

    # Convert the json predictions to the .tags format
    python scripts/convert_predictions_to_streusle2_tags_format.py \
        --predictions-path models/${modelname}/${modelname}_dev_predictions.jsonl \
        --test-data-path ./data/streusle2.0/streusle.tags.dev \
        --output-path models/${modelname}/${modelname}_dev_predictions.jsonl.tags

    # Convert the .tags format file to a .sst file format file
    if [ -f models/${modelname}/${modelname}_dev_predictions.jsonl.tags.sst ] ; then
        rm models/${modelname}/${modelname}_dev_predictions.jsonl.tags.sst
    fi
    python2 scripts/streusle2.0_scripts/tags2sst.py models/${modelname}/${modelname}_dev_predictions.jsonl.tags > models/${modelname}/${modelname}_dev_predictions.jsonl.tags.sst

    # Run mweval
    if [ -f models/${modelname}/${modelname}_dev_predictions.mweval_results ] ; then
        rm models/${modelname}/${modelname}_dev_predictions.mweval_results
    fi
    echo "Running mweval"
    python2 scripts/streusle2.0_scripts/mweval.py \
        data/streusle2.0/streusle.tags.sst.dev \
        models/${modelname}/${modelname}_dev_predictions.jsonl.tags.sst > models/${modelname}/${modelname}_dev_predictions.mweval_results
    # Run ssteval
    if [ -f models/${modelname}/${modelname}dev_predictions.ssteval_results ] ; then
        rm models/${modelname}/${modelname}dev_predictions.ssteval_results
    fi
    echo "Running ssteval"
    python2 scripts/streusle2.0_scripts/ssteval.py \
        data/streusle2.0/streusle.tags.sst.dev \
        models/${modelname}/${modelname}_dev_predictions.jsonl.tags.sst > models/${modelname}/${modelname}dev_predictions.ssteval_results
done
