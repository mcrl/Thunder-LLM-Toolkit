#!/bin/bash

marks=$@
if [ -z "$marks" ]; then
    marks=""
else
    marks="-m $marks"
fi
tests=`cat misc/inference_testlist/testlist.txt`

export CUBLAS_WORKSPACE_CONFIG=:4096:8
pytest --nf --lf $marks -v ${tests[@]}

