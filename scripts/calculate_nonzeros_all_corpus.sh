#
# AGNEWS
#
python3 calculate_nonzeros_for_sparse_attentions.py \
  --corpus agnews \
  --corpus-path "../data/corpus/agnews/test.xml"  \
  --load-model-path "../data/saved-models/test-agnews-softmax/" \
  --load-explainer-path "../data/saved-models/communicate-agnews-softmax/"  \
  --gpu-id 1

python3 calculate_nonzeros_for_sparse_attentions.py \
  --corpus agnews \
  --corpus-path "../data/corpus/agnews/test.xml"  \
  --load-model-path "../data/saved-models/test-agnews-sparsemax/" \
  --load-explainer-path "../data/saved-models/communicate-agnews-sparsemax/"  \
  --gpu-id 1

python3 calculate_nonzeros_for_sparse_attentions.py \
  --corpus agnews \
  --corpus-path "../data/corpus/agnews/test.xml"  \
  --load-model-path "../data/saved-models/test-agnews-entmax15/" \
  --load-explainer-path "../data/saved-models/communicate-agnews-entmax15/"  \
  --gpu-id 1



#
# IMDB
#
python3 calculate_nonzeros_for_sparse_attentions.py \
  --corpus imdb \
  --corpus-path "../data/corpus/imdb/test/"  \
  --load-model-path "../data/saved-models/test-imdb-softmax/" \
  --load-explainer-path "../data/saved-models/communicate-imdb-softmax/"  \
  --gpu-id 1

python3 calculate_nonzeros_for_sparse_attentions.py \
  --corpus imdb \
  --corpus-path "../data/corpus/imdb/test/"  \
  --load-model-path "../data/saved-models/test-imdb-sparsemax/" \
  --load-explainer-path "../data/saved-models/communicate-imdb-sparsemax/"  \
  --gpu-id 1

python3 calculate_nonzeros_for_sparse_attentions.py \
  --corpus imdb \
  --corpus-path "../data/corpus/imdb/test/"  \
  --load-model-path "../data/saved-models/test-imdb-entmax15/" \
  --load-explainer-path "../data/saved-models/communicate-imdb-entmax15/"  \
  --gpu-id 1


#
# SNLI
#
python3 calculate_nonzeros_for_sparse_attentions.py \
  --corpus snli \
  --corpus-path "../data/corpus/snli/snli_1.0_test.jsonl"  \
  --load-model-path "../data/saved-models/test-snli-softmax/" \
  --load-explainer-path "../data/saved-models/communicate-snli-with-emb-softmax/"  \
  --gpu-id 1

python3 calculate_nonzeros_for_sparse_attentions.py \
  --corpus snli \
  --corpus-path "../data/corpus/snli/snli_1.0_test.jsonl"  \
  --load-model-path "../data/saved-models/test-snli-sparsemax/" \
  --load-explainer-path "../data/saved-models/communicate-snli-with-emb-sparsemax/"  \
  --gpu-id 1

python3 calculate_nonzeros_for_sparse_attentions.py \
  --corpus snli \
  --corpus-path "../data/corpus/snli/snli_1.0_test.jsonl"  \
  --load-model-path "../data/saved-models/test-snli-entmax15/" \
  --load-explainer-path "../data/saved-models/communicate-snli-with-emb-entmax15/"  \
  --gpu-id 1



#
# SST
#
python3 calculate_nonzeros_for_sparse_attentions.py \
  --corpus sst \
  --corpus-path "../data/corpus/sst/test.txt"  \
  --load-model-path "../data/saved-models/test-sst-softmax/" \
  --load-explainer-path "../data/saved-models/communicate-sst-softmax/"  \
  --gpu-id 1

python3 calculate_nonzeros_for_sparse_attentions.py \
  --corpus sst \
  --corpus-path "../data/corpus/sst/test.txt"  \
  --load-model-path "../data/saved-models/test-sst-sparsemax/" \
  --load-explainer-path "../data/saved-models/communicate-sst-sparsemax/"  \
  --gpu-id 1

python3 calculate_nonzeros_for_sparse_attentions.py \
  --corpus sst \
  --corpus-path "../data/corpus/sst/test.txt"  \
  --load-model-path "../data/saved-models/test-sst-entmax15/" \
  --load-explainer-path "../data/saved-models/communicate-sst-entmax15/"  \
  --gpu-id 1


#
# YELP
#
python3 calculate_nonzeros_for_sparse_attentions.py \
  --corpus yelp \
  --corpus-path "../data/corpus/yelp/review_test.json"  \
  --load-model-path "../data/saved-models/test-yelp-softmax/" \
  --load-explainer-path "../data/saved-models/communicate-yelp-softmax/"  \
  --gpu-id 1

python3 calculate_nonzeros_for_sparse_attentions.py \
  --corpus yelp \
  --corpus-path "../data/corpus/yelp/review_test.json"  \
  --load-model-path "../data/saved-models/test-yelp-sparsemax/" \
  --load-explainer-path "../data/saved-models/communicate-yelp-sparsemax/"  \
  --gpu-id 1

python3 calculate_nonzeros_for_sparse_attentions.py \
  --corpus yelp \
  --corpus-path "../data/corpus/yelp/review_test.json"  \
  --load-model-path "../data/saved-models/test-yelp-entmax15/" \
  --load-explainer-path "../data/saved-models/communicate-yelp-entmax15/"  \
  --gpu-id 1



# compare
python3 calculate_nonzeros_for_sparse_attentions.py \
  --corpus compare \
  --corpus-path "../data/corpus/compare/devel.txt"  \
  --load-model-path "../data/saved-models/test-compare-sparsemax/" \
  --load-explainer-path "../data/saved-models/communicate-compare-new-sparsemax/"  \
  --gpu-id 0

python3 calculate_nonzeros_for_sparse_attentions.py \
  --corpus compare \
  --corpus-path "../data/corpus/compare/devel.txt"  \
  --load-model-path "../data/saved-models/test-compare-entmax15/" \
  --load-explainer-path "../data/saved-models/communicate-compare-new-entmax15/"  \
  --gpu-id 0
