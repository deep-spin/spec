import json

import nltk
import torchtext

from spec.dataset import fields
from spec.dataset.corpora.corpus import Corpus


class SNLICorpus(Corpus):

    @staticmethod
    def create_fields_tuples():
        tokenizer = nltk.WordPunctTokenizer()
        words_field = fields.WordsField(tokenize=tokenizer.tokenize)
        fields_tuples = [
            ('words', words_field),
            ('words_hyp', words_field),
            ('target', fields.TagsField())
        ]
        return fields_tuples

    def _read(self, file):
        for line in file:
            data = json.loads(line)
            label = data['gold_label']
            premise = data['sentence1']
            hypothesis = data['sentence2']
            if label == '-':
                # These were cases where the annotators disagreed; we'll just
                # skip them. It's like 800 / 500k examples in the train data
                continue
            yield self.make_torchtext_example(premise, hypothesis, label)

    def make_torchtext_example(self, prem, hyp, label):
        ex = {'words': prem, 'words_hyp': hyp, 'target': label}
        if 'target' not in self.fields_dict.keys():
            del ex['target']
        assert ex.keys() == self.fields_dict.keys()
        return torchtext.data.Example.fromdict(ex, self.fields_dict)


if __name__ == '__main__':
    from spec.dataset.corpora.test_corpus import quick_test
    quick_test(
        SNLICorpus,
        '../../../data/corpus/snli/snli_1.0_test.jsonl',
        lazy=True,
    )
