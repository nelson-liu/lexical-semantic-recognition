#!/usr/bin/env bash

cut -d"	" -f-10 $1 > ${1%lex}
