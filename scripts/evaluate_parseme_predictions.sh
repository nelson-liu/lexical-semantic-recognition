#!/usr/bin/env bash

set -e

if [ -z "$1" ]
then
    echo "First command line argument is null"
    echo "Please call the script with: script [parseme_predictions_jsonl_path] [parseme_gold_path]"
    exit 1
fi

if [ -z "$2" ]
then
    echo "Second command line argument is null"
    echo "Please call the script with: script [parseme_predictions_jsonl_path] [parseme_gold_path]"
    exit 1
fi

python3 scripts/convert_predictions_to_parseme_format.py \
    --predictions-path "${1}" \
    --test-data-path "${2}" \
    --output-path "${1}.parseme_format"

echo "Evaluating with PARSEME eval scripts"

python3 scripts/parseme_eval/evaluate.py \
    --gold "${2}" \
    --pred "${1}.parseme_format" >| "${1}.parseme_results"

cat "${1}.parseme_results"
