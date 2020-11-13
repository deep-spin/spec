import torchtext

from spec.dataset import fields
from spec.dataset.corpora.corpus import Corpus


class TextCorpus(Corpus):

    @staticmethod
    def create_fields_tuples():
        fields_tuples = [
            ('words', fields.WordsField())
        ]
        return fields_tuples

    def read(self, texts):
        if self.lazy:
            for ex in self._read(texts):
                yield ex
        else:
            return list(self._read(texts))

    def _read(self, texts):
        self._nb_examples = 0
        if not isinstance(texts, (list, tuple)):
            texts = [texts]
        for text in texts:
            self._nb_examples += 1
            yield self.make_torchtext_example(text)

    def make_torchtext_example(self, text):
        ex = {'words': text}
        return torchtext.data.Example.fromdict(ex, self.fields_dict)


if __name__ == '__main__':
    from spec.dataset.corpora.test_corpus import quick_test
    raw_texts = [
        'Lorem ipsum dolor sit amet, consectetur adipisicing elit',
        'tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim',
        'quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea',
        'consequat. Duis aute irure dolor in reprehenderit in voluptate velit',
        'cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat',
        'proident, sunt in culpa qui officia deserunt mollit anim id est.'
    ]
    quick_test(
        TextCorpus,
        raw_texts,
        lazy=True,
    )
