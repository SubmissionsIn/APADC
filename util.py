import logging
import numpy as np
import math


def get_logger():
    """Get logging."""
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def next_batch(X1, X2, X3, X4, batch_size):
    """Return data for next batch"""
    tot = X1.shape[0]
    total = math.ceil(tot / batch_size)
    for i in range(int(total)):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(tot, end_idx)
        batch_x1 = X1[start_idx: end_idx, ...]
        batch_x2 = X2[start_idx: end_idx, ...]
        batch_x3 = X3[start_idx: end_idx, ...]
        batch_x4 = X4[start_idx: end_idx, ...]

        yield (batch_x1, batch_x2, batch_x3, batch_x4, (i + 1))


def cal_std(logger, *arg):
    """Return the average and its std"""
    if len(arg) == 3:
        print('ACC:'+ str(arg[0]))
        print('NMI:'+ str(arg[1]))
        print('ARI:'+ str(arg[2]))
        output = "ACC {:.3f} std {:.3f} NMI {:.3f} std {:.3f} ARI {:.3f} std {:.3f}".format( np.mean(arg[0]),
                                                                                             np.std(arg[0]),
                                                                                             np.mean(arg[1]),
                                                                                             np.std(arg[1]),
                                                                                             np.mean(arg[2]),
                                                                                             np.std(arg[2]))
    elif len(arg) == 1:
        print(arg)
        output = "ACC {:.3f} std {:.3f}".format(np.mean(arg), np.std(arg))

    print(output)
    return


def normalize(x):
    """Normalize"""
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x
