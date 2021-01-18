#!/usr/bin/env bash

mkdir -p data/embs/glove
cd data/embs/glove || exit

wget -c https://allennlp.s3.amazonaws.com/datasets/glove/glove.840B.300d.txt.gz
gzip -d glove.840B.300d.txt.gz
rm glove.840B.300d.txt.gz
