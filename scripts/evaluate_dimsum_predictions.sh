#!/usr/bin/env bash

set -e

if [ -z "$1" ]
then
    echo "First command line argument is null"
    echo "Please call the script with: script [dimsum_predictions_jsonl_path] [dimsum_gold_path]"
    exit 1
fi

if [ -z "$2" ]
then
    echo "Second command line argument is null"
    echo "Please call the script with: script [dimsum_predictions_jsonl_path] [dimsum_gold_path]"
    exit 1
fi

python3 scripts/convert_predictions_to_dimsum_format.py \
    --predictions-path "${1}" \
    --test-data-path "${2}" \
    --output-path "${1}.dimsum_format"

echo "Evaluating with dimsumeval"

python2.7 scripts/dimsum_eval/dimsumeval.py -C \
    "${2}" \
    "${1}.dimsum_format" >| "${1}.dimsum_results"
cat "${1}.dimsum_results"
