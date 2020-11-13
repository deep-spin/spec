#!/usr/bin/env bash

# create data folder
mkdir -p data/corpus/
cd data/corpus/ || exit

# agnews
mkdir -p agnews
cd agnews || exit
wget -c http://groups.di.unipi.it/~gulli/newsspace200.xml.bz
bzip2 -d newsspace200.xml.bz
rm newsspace200.xml.bz
cd ..

# imdb
wget -c http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xvzf aclImdb_v1.tar.gz
mv aclImdb imdb
rm  aclImdb_v1.tar.gz

# sst
wget -c https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip
unzip trainDevTestTrees_PTB.zip
mv trees sst
rm trainDevTestTrees_PTB.zip

# snli
wget -c https://nlp.stanford.edu/projects/snli/snli_1.0.zip
unzip snli_1.0.zip
mv snli_1.0 snli
rm snli_1.0.zip

# yelp
echo "please, complete the form and download the dataset from here:"
echo "https://www.yelp.com/dataset"
echo "create a folder called 'yelp' and move the 'review.json' file into it"
