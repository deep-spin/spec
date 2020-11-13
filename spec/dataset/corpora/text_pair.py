import torchtext

from spec.dataset import fields
from spec.dataset.corpora.corpus import Corpus


class TextPairCorpus(Corpus):

    @staticmethod
    def create_fields_tuples():
        fields_tuples = [
            ('words', fields.WordsField()),
            ('words_hyp', fields.WordsField())
        ]
        return fields_tuples

    def read(self, texts_ab):
        if self.lazy:
            for ex in self._read(texts_ab):
                yield ex
        else:
            return list(self._read(texts_ab))

    def _read(self, texts_ab):
        self._nb_examples = 0
        texts_a = texts_ab[0]
        texts_b = texts_ab[1]
        if not isinstance(texts_a, (list, tuple)):
            texts_a = [texts_a]
        if not isinstance(texts_b, (list, tuple)):
            texts_b = [texts_b]
        for text_a, text_b in zip(texts_a, texts_b):
            self._nb_examples += 1
            yield self.make_torchtext_example(text_a, text_b)

    def make_torchtext_example(self, prem, hyp):
        ex = {'words': prem, 'words_hyp': hyp}
        return torchtext.data.Example.fromdict(ex, self.fields_dict)
