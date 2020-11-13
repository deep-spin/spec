from pathlib import Path

import numpy as np
import torchtext

from spec.dataset import fields
from spec.dataset.corpora.corpus import Corpus


class IWSLTCorpus(Corpus):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.src_file = None
        self.hyp_file = None
        self.trg_file = None
        self.att_data = None

    @staticmethod
    def create_fields_tuples():
        hyp_target_field = fields.WordsField(lower=True)
        fields_tuples = [
            ('words', fields.WordsField(lower=True)),
            ('words_hyp', hyp_target_field),
            ('target', hyp_target_field),
            ('attn', fields.AttentionField())
        ]
        return fields_tuples

    def read(self, corpus_path):
        if '/grad-' in corpus_path:
            self.src_file = Path(corpus_path + '.src').open('r', encoding='utf8')  # noqa
            self.hyp_file = Path(corpus_path + '.grad_hyp.txt').open('r', encoding='utf8')  # noqa
            self.trg_file = Path(corpus_path + '.trg').open('r', encoding='utf8')  # noqa
            self.att_data = np.load(Path(corpus_path + '.grad.npy'), allow_pickle=True)  # noqa
        else:
            self.src_file = Path(corpus_path + '.src').open('r', encoding='utf8')  # noqa
            self.hyp_file = Path(corpus_path + '.attn_hyp.txt').open('r', encoding='utf8')  # noqa
            self.trg_file = Path(corpus_path + '.trg').open('r', encoding='utf8')  # noqa
            self.att_data = np.load(Path(corpus_path + '.att_att.npy'), allow_pickle=True)  # noqa
        self._current_line = 0
        self._nb_examples = len(self.att_data)
        if self.lazy is True:
            return self
        else:
            return list(self)

    def __iter__(self):
        for ex in self._read(self.src_file,
                             self.hyp_file,
                             self.trg_file,
                             self.att_data):
            self._current_line += 1
            yield ex
        self.start_over()
        self.read_once = True

    def open(self, corpus_path):
        self.closed = False
        return None

    def start_over(self):
        self._current_line = 0
        self.src_file.seek(0)
        self.hyp_file.seek(0)
        self.trg_file.seek(0)

    def close(self):
        self.src_file.close()
        self.hyp_file.close()
        self.trg_file.close()
        self.src_file = None
        self.hyp_file = None
        self.trg_file = None
        self.closed = True

    def __del__(self):
        if self.src_file is not None:
            self.close()

    def _read(self, src_file, hyp_file, trg_file, att_data):
        for i, (src, hyp, tgr, attn) in enumerate(zip(src_file,
                                                      hyp_file,
                                                      trg_file,
                                                      att_data)):
            src = src.strip().split()
            hyp = hyp.strip().split()
            tgr = tgr.strip().split()
            assert np.allclose(attn[len(src)+1:, :len(hyp)], 0)
            yield self.make_torchtext_example(src, hyp, tgr, attn)

    def make_torchtext_example(self, src, hyp, tgr, attn):
        ex = {'words': src, 'words_hyp': hyp, 'target': tgr, 'attn': attn}
        assert ex.keys() == self.fields_dict.keys()
        return torchtext.data.Example.fromdict(ex, self.fields_dict)


if __name__ == '__main__':
    def filter_len(x):
        return True

    fields_tuples = IWSLTCorpus.create_fields_tuples()
    words_field = fields_tuples[0][1]
    words_hyp_field = fields_tuples[1][1]
    target_field = fields_tuples[2][1]
    attn_field = fields_tuples[3][1]

    train_corpus = IWSLTCorpus(fields_tuples, lazy=True)
    print('reading...')
    train_examples = train_corpus.read(
        '../../../data/saved-translation-models/'
        'iwslt-ende-bahdanau-softmax/attn-dev/dev'
    )
    print('ok')

    from spec.dataset.dataset import TextDataset
    from spec.dataset.modules.iterator import LazyBucketIterator
    dataset = TextDataset(train_examples, fields_tuples, filter_pred=filter_len)
    print('BUILD VOCAB')
    print('=====================')
    for _, field in fields_tuples:
        if field.use_vocab:
            field.build_vocab(dataset)

    print('Size of the dataset:', len(dataset))
    print('TESTING DATASET')
    print('=====================')
    for i, ex in enumerate(dataset):
        if i >= 4:
            continue
        print(ex.words, ex.target)
        n1, n2 = len(ex.words), len(ex.target)
        print(ex.attn[1:1+n1, 1:1+n2])  # exclude bos
        # import ipdb; ipdb.set_trace()

    print('Second run:')
    for i, ex in enumerate(dataset):
        if i >= 4:
            continue
        print(ex.words, ex.target)

    print('TESTING ITERATOR')
    print('=====================')
    iterator = LazyBucketIterator(
        dataset=dataset,
        batch_size=2,
        repeat=False,
        sort_key=dataset.sort_key,
        sort=False,
        sort_within_batch=False,
        # shuffle batches
        shuffle=False,
        device=None,
        train=True,
    )
    for i, batch in enumerate(iterator):
        import ipdb; ipdb.set_trace()



