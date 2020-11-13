import torchtext

from spec.dataset import fields
from spec.dataset.corpora.corpus import Corpus


class TextPairWithMarksCorpus(Corpus):

    @staticmethod
    def create_fields_tuples():
        fields_tuples = [
            ('words', fields.WordsField()),
            ('words_hyp', fields.WordsField()),
            ('marks', fields.MarkIndexesField()),
        ]
        return fields_tuples

    def read(self, texts_abc):
        if self.lazy:
            for ex in self._read(texts_abc):
                yield ex
        else:
            return list(self._read(texts_abc))

    def _read(self, texts_abc):
        self._nb_examples = 0
        texts_a = texts_abc[0]
        texts_b = texts_abc[1]
        texts_c = texts_abc[2]
        if not isinstance(texts_a, (list, tuple)):
            texts_a = [texts_a]
        if not isinstance(texts_b, (list, tuple)):
            texts_b = [texts_b]
        if not isinstance(texts_c, (list, tuple)):
            texts_c = [texts_c]
        for text_a, text_b, text_c in zip(texts_a, texts_b, texts_c):
            self._nb_examples += 1
            yield self.make_torchtext_example(text_a, text_b, text_c)

    def make_torchtext_example(self, prem, hyp, marks):
        ex = {'words': prem, 'words_hyp': hyp, 'marks': marks}
        return torchtext.data.Example.fromdict(ex, self.fields_dict)
