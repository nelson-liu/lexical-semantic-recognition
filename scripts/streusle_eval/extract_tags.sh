#!/usr/bin/env bash

grep -v '^#' $1 | cut -d"	" -f4 > ${1%lex}.upos
