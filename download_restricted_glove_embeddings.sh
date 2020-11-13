#!/usr/bin/env bash

mkdir -p data/embs/glove
cd data/embs/glove || exit

wget -c https://www.dropbox.com/s/v0n4futsuz32u7u/glove_restricted_to_vocab.zip?dl=1
unzip glove_restricted_to_vocab.zip
rm glove_restricted_to_vocab.zip
mv glove/* .
rm -rf glove
