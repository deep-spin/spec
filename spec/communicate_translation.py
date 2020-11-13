import logging
from pathlib import Path

from spec import explainers, constants
from spec import iterator
from spec import laymen
from spec import optimizer
from spec import opts
from spec.communicator_translation import CommunicatorTranslation
from spec.dataset import dataset, fields
from spec.dataset.corpora import available_corpora
from spec.models.utils import freeze_all_module_params

logger = logging.getLogger(__name__)


def run(options):
    logger.info('Running with options: {}'.format(options))

    fields_tuples = available_corpora[options.corpus].create_fields_tuples()

    logger.info('Building train corpus: {}'.format(options.train_path))
    train_dataset = dataset.build(options.train_path, fields_tuples, options)

    logger.info('Building train iterator...')
    train_iter = iterator.build(train_dataset,
                                options.gpu_id,
                                options.train_batch_size,
                                is_train=True,
                                lazy=options.lazy_loading)

    dev_dataset = None
    dev_iter = None
    if options.dev_path is not None:
        logger.info('Building dev dataset: {}'.format(options.dev_path))
        dev_dataset = dataset.build(options.dev_path, fields_tuples, options)
        logger.info('Building dev iterator...')
        dev_iter = iterator.build(dev_dataset,
                                  options.gpu_id,
                                  options.dev_batch_size,
                                  is_train=False,
                                  lazy=options.lazy_loading)

    test_dataset = None
    test_iter = None
    if options.test_path is not None:
        logger.info('Building test dataset: {}'.format(options.test_path))
        test_dataset = dataset.build(options.test_path, fields_tuples, options)
        logger.info('Building test iterator...')
        test_iter = iterator.build(test_dataset,
                                   options.gpu_id,
                                   options.dev_batch_size,
                                   is_train=False,
                                   lazy=options.lazy_loading)

    datasets = [train_dataset, dev_dataset, test_dataset]
    datasets = list(filter(lambda x: x is not None, datasets))

    # BUILD
    if not options.load:
        logger.info('Building vocabulary...')
        fields.build_vocabs(fields_tuples, train_dataset, datasets, options)

    # OR LOAD
    else:
        logger.info('Loading vocabularies...')
        fields.load_vocabs(options.load, fields_tuples)
        logger.info('Loading vectors...')
        vectors = fields.load_vectors(options)
        if vectors is not None:
            train_dataset.fields['words'].vocab.load_vectors(vectors)

    # STATS
    logger.info('Number of training examples: {}'.format(len(train_dataset)))
    if dev_dataset:
        logger.info('Number of dev examples: {}'.format(len(dev_dataset)))
    if test_dataset:
        logger.info('Number of test examples: {}'.format(len(test_dataset)))
    for name, field in fields_tuples:
        if field.use_vocab:
            logger.info('{} vocab size: {}'.format(name, len(field.vocab)))

    # BUILD COMMUNICATION
    if not options.load_communication:
        logger.info('Building explainer...')
        explainer = explainers.build(options, fields_tuples, None)
        logger.info('Building explainer optimizer...')
        explainer_optim = optimizer.build(options, explainer.parameters())

        logger.info('Building layman...')
        msg_size = explainer.get_output_size()
        layman = laymen.build(options, fields_tuples, msg_size, None)
        logger.info('Building layman optimizer...')
        layman_optim = optimizer.build(options, layman.parameters())

    # OR LOAD COMMUNICATION
    else:
        logger.info('Loading explainer...')
        explainer = explainers.load(
            options.load_communication,
            fields_tuples,
            options.gpu_id
        )
        logger.info('Loading explainer optimizer...')
        explainer_optim = optimizer.load(
            options.load_communication,
            explainer.parameters(),
            name=constants.EXPLAINER_OPTIMIZER,
            config_name=constants.COMMUNICATION_CONFIG
        )

        logger.info('Loading layman...')
        msg_size = explainer.get_output_size()
        layman = laymen.load(
            options.load_communication,
            fields_tuples,
            msg_size,
            options.gpu_id
        )
        logger.info('Loading layman optimizer...')
        layman_optim = optimizer.load(
            options.load_communication,
            layman.parameters(),
            name=constants.LAYMAN_OPTIMIZER,
            config_name=constants.COMMUNICATION_CONFIG
        )

    logger.info('Explainer info: ')
    logger.info(str(explainer))
    logger.info('Explainer optimizer info: ')
    logger.info(str(explainer_optim))
    logger.info('Layman info: ')
    logger.info(str(layman))
    logger.info('Layman optimizer info: ')
    logger.info(str(layman_optim))

    if options.freeze_explainer_params:
        logger.info('Freezing explainer params...')
        freeze_all_module_params(explainer)

    # TRAIN
    logger.info('Building trainer...')
    communicator = CommunicatorTranslation(
        train_iter,
        explainer,
        layman,
        explainer_optim,
        layman_optim,
        options,
        dev_iter=dev_iter,
        test_iter=test_iter
    )

    # resume training from a checkpoint
    if options.resume_epoch and options.load is None:
        logger.info('Resuming communication...')
        communicator.resume(options.resume_epoch)

    # train the communication
    communicator.train()

    if options.save_explanations:
        logger.info('Saving explanations to {}'.format(options.save_explanations))
        # save explanations (run with 0 epochs to ignore the communication)
        ds_iterator = test_iter if test_iter is not None else dev_iter
        communicator.save_explanations(
            options.save_explanations, ds_iterator,
            max_explanations=options.max_explanations
        )

    # SAVE
    if options.save:
        logger.info('Saving path: {}'.format(options.save))
        config_path = Path(options.save)
        config_path.mkdir(parents=True, exist_ok=True)

        # save communication modules
        logger.info('Saving communication config options...')
        opts.save(config_path, options, name=constants.COMMUNICATION_CONFIG)
        logger.info('Saving explainer...')
        explainers.save(config_path, explainer)
        logger.info('Saving layman...')
        laymen.save(config_path, layman)
        logger.info('Saving explainer optimizer...')
        optimizer.save(
            config_path, explainer_optim, name=constants.EXPLAINER_OPTIMIZER
        )
        logger.info('Saving layman optimizer...')
        optimizer.save(
            config_path, layman_optim, name=constants.LAYMAN_OPTIMIZER
        )
