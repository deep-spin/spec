import logging
from pathlib import Path

from spec import constants
from spec import iterator
from spec import models
from spec.dataset import dataset, fields
from spec.dataset.corpora import available_corpora
from spec.predicter import Predicter

logger = logging.getLogger(__name__)


def run(options):
    fields_tuples = available_corpora[options.corpus].create_fields_tuples()
    # fields_tuples += features.load(options.load)

    if options.test_path is None and options.text is None:
        raise Exception('You should inform a path to test data or a text.')

    if options.test_path is not None and options.text is not None:
        raise Exception('You cant inform both a path to test data and a text.')

    dataset_iter = None

    if options.test_path is not None and options.text is None:
        logger.info('Building test dataset: {}'.format(options.test_path))
        test_tuples = list(filter(lambda x: x[0] != 'target', fields_tuples))
        test_dataset = dataset.build(options.test_path, test_tuples, options)

        logger.info('Building test iterator...')
        dataset_iter = iterator.build(test_dataset,
                                      options.gpu_id,
                                      options.dev_batch_size,
                                      is_train=False,
                                      lazy=options.lazy_loading)

    if options.text is not None and options.test_path is None:
        logger.info('Preparing text...')
        test_tuples = list(filter(lambda x: x[0] != 'target', fields_tuples))
        test_dataset = dataset.build_texts(options.text, test_tuples, options)

        logger.info('Building iterator...')
        dataset_iter = iterator.build(test_dataset,
                                      options.gpu_id,
                                      options.dev_batch_size,
                                      is_train=False,
                                      lazy=options.lazy_loading)

    logger.info('Loading vocabularies...')
    fields.load_vocabs(options.load, fields_tuples)

    logger.info('Loading model...')
    model = models.load(options.load, fields_tuples, options.gpu_id)

    logger.info('Predicting...')
    predicter = Predicter(dataset_iter, model)
    predictions = predicter.predict(options.prediction_type)

    logger.info('Preparing to save...')
    if options.prediction_type == 'classes':
        target_field = dict(fields_tuples)['target']
        prediction_target = transform_classes_to_target(target_field,
                                                        predictions)
        predictions_str = transform_predictions_to_text(prediction_target)
    else:
        predictions_str = transform_predictions_to_text(predictions)

    if options.test_path is not None:
        save_predictions(
            options.output_dir,
            predictions_str,
        )
    else:
        logger.info(options.text)
        logger.info(predictions_str)

    return predictions


def save_predictions(directory, predictions_str):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    output_path = Path(directory, constants.PREDICTIONS)
    save_predictions_in_a_file(output_path, predictions_str)
    logger.info('Predictions saved in {}'.format(output_path))


def save_predictions_in_a_file(output_file_path, predictions_str):
    output_file_path.write_text(predictions_str + '\n')


def save_predictions_in_a_dir(ourpur_dir_path, file_names, predictions_str):
    assert ourpur_dir_path.is_dir()
    predictions_for_each_file = predictions_str.split('\n')
    for f_name, pred_str in zip(file_names, predictions_for_each_file):
        output_path = Path(ourpur_dir_path, f_name)
        output_path.write_text(pred_str + '\n')


def transform_classes_to_target(target_field, predictions):
    tagged_predicitons = []
    for preds in predictions:
        target_preds = [target_field.vocab.itos[c] for c in preds]
        tagged_predicitons.append(target_preds)
    return tagged_predicitons


def transform_predictions_to_text(predictions):
    text = []
    is_prob = isinstance(predictions[0][0], list)
    for pred in predictions:
        sentence = []
        for p in pred:
            if is_prob:
                sentence.append(', '.join(['%.8f' % c for c in p]))
            else:
                sentence.append(p)
        if is_prob:
            text.append(' | '.join(sentence))
        else:
            text.append(' '.join(sentence))
    return '\n'.join(text)
