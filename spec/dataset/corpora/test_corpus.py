def quick_test(corpus_cls, corpus_path, lazy=True, **kwargs):
    from spec.dataset import fields
    from spec.dataset.dataset import TextDataset
    from spec.dataset.modules.iterator import LazyBucketIterator

    def filter_len(x):
        return True

    fields_tuples = corpus_cls.create_fields_tuples()
    words_field = fields_tuples[0][1]
    target_field = fields_tuples[-1][1]

    train_corpus = corpus_cls(fields_tuples, lazy=lazy, **kwargs)
    train_examples = train_corpus.read(corpus_path)
    # print(train_corpus.nb_examples)  # raise exception
    labels_count = {}
    avg_len = 0
    max_len = 0
    min_len = 999999
    for i, ex in enumerate(train_examples):
        if ex.target[0] not in labels_count:
            labels_count[ex.target[0]] = 0
        labels_count[ex.target[0]] += 1
        # if i > 4:
        #     continue
        avg_len += len(ex.words)
        max_len = max(max_len, len(ex.words))
        min_len = min(min_len, len(ex.words))
        # print(ex.words, ex.target)
    print(labels_count)
    print('min len:', min_len)
    print('max len:', max_len)
    print('avg len:', avg_len / len(train_examples))
    print('nb examples:', len(train_examples))
    print('')

    for i, ex in enumerate(train_examples):
        if i > 4:
            continue
        print(ex.words, ex.target)

    print('\n\n\n')

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
        if i >= 2:
            continue
        tags = [target_field.vocab.itos[t] for x in batch.target.tolist() for t in x]
        words = [[words_field.vocab.itos[y] for y in x] for x in batch.words]
        for x, y in zip(tags, words):
            print(x, y)
    print('Second run:')
    for i, batch in enumerate(iterator):
        if i >= 2:
            continue
        tags = [target_field.vocab.itos[t] for x in batch.target.tolist() for t in x]
        words = [[words_field.vocab.itos[y] for y in x] for x in batch.words]
        for x, y in zip(tags, words):
            print(x, y)
