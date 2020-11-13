import json

import nltk
import torchtext

from spec.dataset import fields
from spec.dataset.corpora.corpus import Corpus


class YelpCorpus(Corpus):

    @staticmethod
    def create_fields_tuples():
        tokenizer = nltk.WordPunctTokenizer()
        fields_tuples = [
            ('words', fields.WordsField(tokenize=tokenizer.tokenize)),
            ('target', fields.TagsField())
        ]
        return fields_tuples

    def _read(self, file):
        for line in file:
            data = json.loads(line.strip())
            text = data['text']
            label = str(int(data['stars']))
            example = self.make_torchtext_example(text, label)
            yield example

    def make_torchtext_example(self, text, label=None):
        ex = {'words': text, 'target': label}
        if 'target' not in self.fields_dict.keys():
            del ex['target']
        assert ex.keys() == self.fields_dict.keys()
        return torchtext.data.Example.fromdict(ex, self.fields_dict)


if __name__ == '__main__':
    from spec.dataset.corpora.test_corpus import quick_test
    quick_test(
        YelpCorpus,
        '../../../data/corpus/yelp/review_train.json',
        lazy=True,
    )
