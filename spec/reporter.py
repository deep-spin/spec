import logging

from spec.stats import BestValueEpoch

logger = logging.getLogger(__name__)


def get_line_bar(template_head):
    line_head = list('-' * len(template_head.strip()))
    bar_indexes = [i for i, c in enumerate(template_head) if c == '|']
    for i in bar_indexes:
        line_head[i] = '+'
    return ''.join(line_head)


class Reporter:
    """
    Simple class to print stats on the screen using logger.info and
    optionally, tensorboard.

    Args:
        output_dir (str): Path location to save tensorboard artifacts.
        use_tensorboard (bool): Whether to log stats on tensorboard server.
        for_communication (bool) Whether it to report communication-only stats:
            TVD, ACC for classifier, ACC for layman. Default is False.
    """

    def __init__(self, output_dir, use_tensorboard, for_communication=False):
        self.tb_writer = None
        if use_tensorboard:
            logger.info('Starting tensorboard logger...')
            logger.info('Type `tensorboard --logdir runs/` in your terminal '
                        'to see live stats.')
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(output_dir)
        self.mode = None
        self.epoch = None
        self.output_dir = output_dir
        self.for_communication = for_communication
        self.template_head = 'Loss    (val / epoch) | '
        self.template_head += 'Prec.     '
        self.template_head += 'Rec.    '
        self.template_head += 'F1     (val / epoch) | '
        self.template_head += 'ACC    (val / epoch) | '
        self.template_head += 'MCC    (val / epoch) | '
        if for_communication:
            self.template_head += 'TVD    (val / epoch) | '
            self.template_head += 'ACC L  | '
            self.template_head += 'ACC C  | '
        self.template_line = get_line_bar(self.template_head)
        self.template_body = '{:7.4f} ({:.4f} / {:2d}) |'  # loss (best/best)
        self.template_body += '{:7.4f}  '  # prec.
        self.template_body += '{:7.4f}  '  # rec.
        self.template_body += '{:7.4f} ({:.4f} / {:2d}) |'  # F1 (best/best)
        self.template_body += '{:7.4f} ({:.4f} / {:2d}) |'  # ACC (best/best)
        self.template_body += '{:7.4f} ({:.4f} / {:2d}) |'  # MCC (best/best)
        if for_communication:
            self.template_body += '{:7.4f} ({:.4f} / {:2d}) |'  # TVD
            self.template_body += '{:7.4f} |'  # ACC L
            self.template_body += '{:7.4f} |'  # ACC C
        self.template_footer = '---'

    def set_mode(self, mode):
        self.mode = mode

    def set_epoch(self, epoch):
        self.epoch = epoch

    def show_head(self):
        logger.info(self.template_head)
        logger.info(self.template_line)

    def show_footer(self):
        logger.info(self.template_footer)

    def show_stats(self, stats_dict, epoch=None):
        communication_stats = []
        if self.for_communication:
            communication_stats = [
                stats_dict['tvd'],
                stats_dict['best_tvd'].value,
                stats_dict['best_tvd'].epoch,
                stats_dict['acc_l'],
                stats_dict['acc_c'],
            ]
        text = self.template_body.format(
            stats_dict['loss'],
            stats_dict['best_loss'].value,
            stats_dict['best_loss'].epoch,
            stats_dict['prec_rec_f1'][0],
            stats_dict['prec_rec_f1'][1],
            stats_dict['prec_rec_f1'][2],
            stats_dict['best_prec_rec_f1'].value[2],
            stats_dict['best_prec_rec_f1'].epoch,
            stats_dict['acc'],
            stats_dict['best_acc'].value,
            stats_dict['best_acc'].epoch,
            stats_dict['mcc'],
            stats_dict['best_mcc'].value,
            stats_dict['best_mcc'].epoch,
            *communication_stats
        )
        if epoch is not None:
            text += '< Ep. {}'.format(epoch)
        logger.info(text)

    def report_progress(self, i, nb_iters, loss):
        print('Loss ({}/{}): {:.4f}'.format(i, nb_iters, loss), end='\r')
        if self.tb_writer is not None:
            j = (self.epoch - 1) * nb_iters + i
            mode_metric = '{}/{}'.format(self.mode, 'moving_loss')
            self.tb_writer.add_scalar(mode_metric, loss, j)

    def report_stats(self, stats_dict):
        self.show_head()
        self.show_stats(stats_dict)
        self.show_footer()
        if self.tb_writer is not None:
            for metric, value in stats_dict.items():
                if isinstance(value, BestValueEpoch):
                    continue
                if metric == 'prec_rec_f1':
                    mm_0 = '{}/{}'.format(self.mode, 'precision')
                    mm_1 = '{}/{}'.format(self.mode, 'recall')
                    mm_2 = '{}/{}'.format(self.mode, 'f1')
                    self.tb_writer.add_scalar(mm_0, value[0], self.epoch)
                    self.tb_writer.add_scalar(mm_1, value[1], self.epoch)
                    self.tb_writer.add_scalar(mm_2, value[2], self.epoch)
                else:
                    mode_metric = '{}/{}'.format(self.mode, metric)
                    self.tb_writer.add_scalar(mode_metric, value, self.epoch)

    def report_stats_history(self, stats_history, start=1):
        self.show_head()
        for i, stats_dict in enumerate(stats_history, start=start):
            self.show_stats(stats_dict, epoch=i)
        self.show_footer()

    def close(self):
        if self.tb_writer is not None:
            # all_scalars_path = Path(self.output_dir, 'all_scalars.json')
            # self.tb_writer.export_scalars_to_json(str(all_scalars_path))
            self.tb_writer.close()
