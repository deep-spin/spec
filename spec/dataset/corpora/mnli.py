from spec.dataset.corpora.snli import SNLICorpus


class MNLICorpus(SNLICorpus):
    pass


if __name__ == '__main__':
    from spec.dataset.corpora.test_corpus import quick_test
    quick_test(
        MNLICorpus,
        '../../../data/corpus/mnli/multinli_1.0_dev_matched.jsonl',
        lazy=True,
    )
