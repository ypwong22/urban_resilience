import numpy as np


def identify_high_extremes(vector, thres):
    thres = np.percentile(vector, thres)
    is_extreme = vector >= thres
    return is_extreme


def identify_low_extremes(vector, thres):
    thres = np.percentile(vector, thres)
    is_extreme = vector <= thres
    return is_extreme


def get_blocks(bool_mask):
    """ Find start & end indices of True's
    """
    bool_mask = bool_mask.astype(int)
    start = np.where(np.diff(np.insert(bool_mask, 0, 0)) == 1)[0]
    end   = np.where(np.diff(np.append(bool_mask, 0)) == -1)[0]
    return start, end


def get_events(is_extreme, tvec):

    def _join_blocks(start, end):
        """ Merge the start-end pairs that are separated by <= 3 months """
        if len(start) == 0:
            return start, end

        is_first_half = np.append((start[1:] - end[:-1]) <= 3, False)
        ind_first_half = np.where(is_first_half)[0]
        ind_second_half = ind_first_half + 1
        is_second_half = np.full(len(is_first_half), False)
        is_second_half[ind_second_half] = True
        return start[~is_second_half], end[~is_first_half]

    start, end = _join_blocks(*get_blocks(is_extreme))

    return tvec[start], tvec[end]


""" Test

import numpy as np
from utils.extremes import *

bool_mask_list = [
    np.array([True]),
    np.array([False]),
    np.array([True, False]),
    np.array([False, True]),
    np.array([True, True, False]),
    np.array([False, True, False]),
    np.array([False, True, True]),
    np.array([False, False, True]),
    np.array([False, True, True, False, False, True, False]),
    np.array([False, True, False, False, False, True, False])
]

for bool_mask in bool_mask_list:
    print(join_blocks(*get_blocks(bool_mask)))
"""