import argparse
import logging

from spec import communicate, communicate_translation
from spec import config_utils
from spec import opts
from spec import predict
from spec import train

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='SpEC')
parser.add_argument('task',
                    type=str,
                    choices=['train',
                             'predict',
                             'communicate',
                             'communicate_translation']
                    )
opts.general_opts(parser)
opts.preprocess_opts(parser)
opts.model_opts(parser)
opts.train_opts(parser)
opts.communication_opts(parser)
opts.predict_opts(parser)


if __name__ == '__main__':
    options = parser.parse_args()
    options.output_dir = config_utils.configure_output(options.output_dir)
    config_utils.configure_logger(options.debug, options.output_dir)
    config_utils.configure_seed(options.seed)
    config_utils.configure_device(options.gpu_id)
    logger.info('Output directory is: {}'.format(options.output_dir))

    if options.task == 'train':
        train.run(options)
    elif options.task == 'predict':
        predict.run(options)
    elif options.task == 'communicate':
        communicate.run(options)
    elif options.task == 'communicate_translation':
        communicate_translation.run(options)
