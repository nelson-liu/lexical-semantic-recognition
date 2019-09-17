#!/usr/bin/env bash
set -x

REF="$1"
shift
for PRED in "$@"; do
    PREFIX="$(basename "$PRED" .jsonl)".
    python streusle_set_lextag.py "$REF" "$PRED" > "$PREFIX"json || exit 1
    python -m json2conllulex "$PREFIX"json > "$PREFIX"conllulex || exit 1
    python -m conllulex2UDlextag "$PREFIX"conllulex > "$PREFIX"UDlextag || exit 1
    python -m UDlextag2json --no-validate-pos --no-validate-type "$PREFIX"UDlextag > "$PREFIX"autoid.json || exit 1
    #rm -f "$PREFIX"json "$PREFIX"conllulex "$PREFIX"UDlextag
done
python -m streuseval "$REF" *.autoid.json
