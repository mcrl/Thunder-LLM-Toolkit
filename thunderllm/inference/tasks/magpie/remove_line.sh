#!/bin/bash

function file_routine(){
    fpath=$1
    # remove 'group_alias' line
    sed -i '/group_alias/d' $fpath
}

filepaths=`file-path`
for f in $filepaths; do
    echo $f
    file_routine $f
    # break
done