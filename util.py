import numpy as np
from sklearn import preprocessing
import itertools
from scipy.special import binom
import functools


def gen_labels(inliers, outliers):
    return np.hstack((np.zeros(inliers), np.ones(outliers)))


def dicretize(data):
    preprocessing.KBinsDiscretizer(n_bins=[3, 2, 2], encode='ordinal').fit(data)


def positivize(data):
    data = (data * .5) + .5
    data = np.where(data > 1, 1, data)
    data = np.where(data < 0, 0, data)
    return data


def normalize(train, test, onlypositive=True):
    max_abs_scaler = preprocessing.MaxAbsScaler()
    standard_scaler = preprocessing.StandardScaler(with_std=False)
    train = max_abs_scaler.fit_transform(standard_scaler.fit_transform(train))
    test = max_abs_scaler.transform(standard_scaler.transform(test))
    if onlypositive:
        train = (train * .5) + .5
        test = (test * .5) + .5
        test = np.where(test > 1, 1, test)
        test = np.where(test < 0, 0, test)
    return train, test, lambda data: positivize(max_abs_scaler.transform(standard_scaler.transform(data)))


def product_iter(ar):
    for a in ar:
        yield np.array(list(itertools.product(*a)))


def corner_combinations(dim, degree):
    return binom(dim, degree) * 2**degree


def corner_combinations_up_to_degree(dim, degree):
    total_comb = 0
    for d in range(1, degree+1):
        total_comb = total_comb + corner_combinations(dim, d)
    return total_comb


def is_in_scaled_range(border_value):
    return (border_value > 0) & (border_value < 1)


def valid_borders(cell_borders):
    for border_pair in cell_borders:
        yield list(filter(is_in_scaled_range, border_pair))


def switch_combinations(switches):
    for switch in switches:
        yield np.array(list(itertools.product(*switch)))


def lattice_dist(degree, depth):
    return degree*depth


UNPACK_MAX = 32


def degree_from_corner_idx(idx, dim):
    total_comb = 0
    for d in itertools.count(1):
        total_comb = total_comb + corner_combinations(dim, degree=d)
        if total_comb > idx:
            return d


def dim_depth_combinations(dim, depth):
    # 3, 2 -> ([2 0 0 ], [1 1 0], [1 0 1], [0 2 0], [0 1 1], [0 0 2]
    # 3, 3 -> ([3 0 0] [2 1 0], [2 0 1], [1 2 0] [1 1 1] [1 0 2] [0 3 0] ...
    return np.array(list(itertools.combinations_with_replacement(np.eye(dim), depth))).sum(axis=1)


def degree_depth_combinations(degree, depth):
    # filters out all from dim_depth_combinations that have one zero
    # so that degree is conformed to
    return np.array(list(itertools.filterfalse(lambda row: (row == 0).any(), dim_depth_combinations(degree, depth))))


def ordinal_masks_for_combination_idx(idx, dim, depth):
    degree = degree_from_corner_idx(idx, dim)

    # Depth gives the distance of neighbor cells to consider
    # degree is the corner/border degree

    no_up_down_combinations = 2 ** degree

    rel_index = idx - corner_combinations_up_to_degree(dim, degree-1)
    # this is the pos for the unit matrix combinations where we want to change ordinals
    pos_where_to_change = int(rel_index / no_up_down_combinations)
    where_to_change_bits = np.array(list(itertools.islice(itertools.combinations(np.eye(dim), degree),
                                                          pos_where_to_change, pos_where_to_change + 1))).sum(axis=1)
    where_to_change_bits_idxs = where_to_change_bits[0].nonzero()[0]

    up_down_combination = np.unpackbits(np.array([rel_index % no_up_down_combinations], dtype=">i4").view(np.uint8))
    up_down_combination = up_down_combination[-degree:UNPACK_MAX]

    # per up down combination we need to consider all combinations for the depth
    # the total sum of changed ordinals must be = depth
    # e.g. up-up (degree 2) depth 3 ->  +2 +1, +1 +2
    step_combinations = degree_depth_combinations(degree, depth)

    up_down_combination = np.where(up_down_combination == 0, -1, up_down_combination)

    # 0 indicate stepping down, so multiply these with -1 (rest stays positive)
    step_combinations_directions = step_combinations * up_down_combination

    no_step_combinations = step_combinations.shape[0]

    for comb in range(no_step_combinations):
        result = np.zeros(dim)
        for j, ordinal in enumerate(where_to_change_bits_idxs):
            result[ordinal] = step_combinations_directions[comb][j]
        yield result


def get_cell_corner_order(o, cell_borders, depth=1):
        """
            Given a point o and cell borders will return the sequence from lowest to
            highest distance (l1) for each of the d**2 corners.

            Parameters
            ----------
            o : ndarray
            A point in the data space.

            cell_borders : ndarray
            borders of a cell, two for each dimension.

            depth : int
            How many neighboring cells should be considered.

            Returns
            -------
            order : ndarray
                A 1-D array that holds a permutation of 0..d**2.
                Values can be thought of as decimal representations of the corners when
                each corner is encoded by a d bit vector (lower border 0, upper border 1).
        """
        bin_width = cell_borders[0][1] - cell_borders[0][0]
        # replace invalid border with nan so that their distance is nan
        cell_borders = np.where(is_in_scaled_range(cell_borders), cell_borders, float('nan'))
        cell_dist = np.abs(np.transpose(cell_borders)-o)
        cell_dist = np.dstack(cell_dist)[0]

        corner_dist = np.empty(0)

        for degree in range(1, depth + 1):
            # the dist_remainer is the distance remaining, when depth > degree.
            # E.g. when degree = 1 and depth = 3, we have one portion of the distance within the cell
            # and the remainer are bin_width * 2. Conversely for degree = 3 and depth = 3 we will
            #  only look at direct borders of degree 3 and do not go deeper, so remainder = 0
            dist_remainer = (depth - degree) * bin_width

            # combinations of dimensions to be switched. Returns array of (D over degree)
            # the ordering is: 12, 13, 23 for three dimensions, degree two
            cell_dist_switches = np.array(list(itertools.combinations(cell_dist, degree)))

            # each switch now has two combinations (either increase or decrease the coordinate)
            # e.g. for 12: 1 down 2 down, 1 down 2 up, 1 up 2 down, 1 down 2 down
            # we need to determine which is the distance order

            # entry for each corner and the distances that need to be traversed for the dimensions
            degree_corner_dist = np.array(list(switch_combinations(cell_dist_switches)))\
                .reshape(-1, degree).sum(axis=1) + dist_remainer
            corner_dist = np.hstack((corner_dist, degree_corner_dist))
            # corner_dist_combinations = np.array(list(itertools.product(*cell_dist)))

        return np.argsort(corner_dist), corner_dist
