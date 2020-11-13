import math

import torch

from spec import constants
from spec.dataset.corpora import available_corpora
from spec.models import LinearBoW


class LinearTfIdf(LinearBoW):
    """
    Linear TF-IDF.
    """

    def __init__(self, fields_tuples, options):
        super().__init__(fields_tuples, options)

        self.idf = torch.zeros(self.words_vocab_size)
        if options.gpu_id is not None:
            self.idf = self.idf.to(options.gpu_id)
        self.nb_documents = 0
        corpus_cls = available_corpora[options.corpus]
        dummy_fields_tuples = corpus_cls.create_fields_tuples()
        corpus = corpus_cls(dummy_fields_tuples, lazy=True)
        examples = corpus.read(options.train_path)
        for ex in examples:
            self.nb_documents += 1
            for w in set(ex.words):
                w_id = self.fields_dict['words'].vocab.stoi[w]
                self.idf[w_id] += 1
        div = torch.div(self.nb_documents, self.idf)
        assert(torch.all(div >= 1))
        self.idf = torch.log(div)
        self.idf[self.idf == float('inf')] = 1e-5  # smoothing
        self.idf = self.idf.unsqueeze(0)
        corpus.close()

    def get_bow(self, words):
        bow = super().get_bow(words)
        tfidf = bow * self.idf.unsqueeze(1)
        return tfidf
