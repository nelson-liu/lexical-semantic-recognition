#!/usr/bin/env bash
set -x

REF="$1"
shift
grep -v '^#' $REF | cut -d"	" -f4 > ${REF%lex}.upos
for PRED in "$@"; do
    PREFIX="${PRED%jsonl}"
    python scripts/streusle_eval/streusle_set_lextag.py "$REF" "$PRED" > "$PREFIX"json || exit 1
    python -m json2conllulex "$PREFIX"json > "$PREFIX"conllulex || exit 1
    python -m conllulex2UDlextag "$PREFIX"conllulex > "$PREFIX"UDlextag || exit 1
    python -m UDlextag2json --no-validate-pos --no-validate-type "$PREFIX"UDlextag > "$PREFIX"autoid.json || exit 1
    python -m json2conllulex "$PREFIX"autoid.json > "$PREFIX"autoid.conllulex
    #rm -f "$PREFIX"json "$PREFIX"conllulex "$PREFIX"UDlextag
    grep -v '^#' "$PREFIX"conllulex | cut -d"	" -f4 > "$PREFIX"conllu.upos || continue
    paste "$PREFIX"conllu.upos ${REF%lex}.upos > "$PREFIX"conllu.upos.predicted_actual
    python -m streuseval "$REF" "$PREFIX"autoid.json > "$PREFIX"streusle_results
    python -m streuseval "$REF" "$PREFIX"autoid.json -x > "$PREFIX"streusle_extended_results
done
