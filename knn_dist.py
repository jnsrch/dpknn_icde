from sklearn.neighbors import KDTree
from sklearn import preprocessing, neighbors
import numpy as np
import itertools
import warnings
import util
import math
warnings.filterwarnings("ignore", category=FutureWarning)


class Knn:
    def __init__(self, data, leaf_size=40):
        self.tree = KDTree(data, leaf_size=leaf_size)
        self.data = data

    def query(self, o, k, weighted=False):
        distances, _ = self.tree.query(o, k=k)

        if weighted:
            return np.sum(distances, axis=1)
        else:
            return distances[:, -1]


class BinnedKnn(Knn):
    def __init__(self, bins_per_dim, data, leaf_size=40, seed=42):
        self.bins_per_dim = bins_per_dim
        self.number_of_dimensions = data.shape[1]
        self.number_of_borders = self.number_of_dimensions * 2

        self.discretizer = preprocessing. \
            KBinsDiscretizer(n_bins=[bins_per_dim] * self.number_of_dimensions, encode='ordinal',
                             strategy='uniform').fit(data)
        self.data_binned = self.discretizer.inverse_transform(self.discretizer.transform(data))
        self.celled_data = self.discretizer.transform(data)

        self.cell_frequencies = np.unique(self.celled_data, axis=0, return_counts=True)

        #l1metric = neighbors.DistanceMetric.get_metric('manhatten')
        self.bin_tree = KDTree(self.data_binned, leaf_size=leaf_size, metric='l1')
        self.PENALTY = self.number_of_dimensions + 1
        self.rnd_generator = np.random.default_rng(seed)

        super().__init__(data, leaf_size)

    def query_cell_frequency(self, cell):
        indexarr = np.argwhere((self.cell_frequencies[0] == cell).all(axis=1))
        if len(indexarr) == 0:
            return 0
        else:
            itemindex = indexarr[0][0]
        return self.cell_frequencies[1][itemindex]

    def get_point_cell_borders(self, o_cell):
        """
            Gives a cell's borders (bin edges).

            Parameters
            ----------
            o_cell : a cell corresponding to a point transformed by the discretizer
                The arrays must have the same shape along all but the third axis.
                1-D or 2-D arrays must have the same shape.

            Returns
            -------
            borders : ndarray
                A 2-D array containing in each row the lower and upper bound for
                each dimension with respect to the cell.
        """
        o_cell = o_cell.astype("int")
        # bk.discretizer.bin_edges_[0][[np.array((0,0))]] leads to Future Warning - but ignore
        lb_o = self.discretizer.bin_edges_[0][[o_cell]] # supposing standardized cells (equal range width)
        ub_o = self.discretizer.bin_edges_[0][[o_cell+1]] # incrementing each array entry for upper bound
        return np.dstack((lb_o,ub_o))[0]

    def get_cell_border_order(self, o, cell_borders):
        """
            Given a point o and cell borders will return the sequence from lowest to
            highest distance for each of the 2*d borders (lower and upper edge).

            Parameters
            ----------
            o : a point in the data space
            cell_borders : borders of a cell, two for each dimension.

            Returns
            -------
            order : ndarray
                A 1-D array that holds a permutation of 0..2d-1.
                values i >= d indicate upper borders for the dimension i%d
                values i < d indicate lower borders
        """
        cell_dist = np.abs(cell_borders-o.reshape(self.number_of_dimensions, 1))\
            .reshape(1, 2 * self.number_of_dimensions)
        order = np.argsort(cell_dist)
        return order

    def is_cell_ordinal_in_range(self, ordinal):
        """
            Will check whether a cell or cell_ordinal is valid.

            Parameters
            ----------
            ordinal : int or cell_array : ndarray
            an ordinal of a cell (as would be given by discretizer.transform())

            Returns
            -------
            boolean
                Whether the cell is valid or out of range (outside the data hyper cube).
        """
        if np.isscalar(ordinal):
            ordinal = np.array([ordinal])

        return ((ordinal < self.number_of_dimensions) & (ordinal >= 0)).all()

    def query(self, o, k, laplace_noise_sd=0, weighted=True, maxdepth=3):
        """
            Returns the l1 lattice neighbor distance. When noidy, then query until gathered elements > k.
            Only considers adjacent cells.

            Parameters
            ----------
            o : ndarray
            A point in the data space, for which we want the lattice distance to neighbor cells depending on k

            k : int
            The number of cell frequencies desired (lower bound when noisy)

            laplace_noise_sd: float
            Magnitude of laplace noise (standard deviation)

            weighted=True : boolean
            Whether to return the distances summed and weighted by number of elements per cell
             or only return final distance to cell which exceeded threshold k.

            Returns
            -------
            dist : int
                L1 distance in unit according to cell width.
        """

        trav = BinnedKnnTraversal(self, o, maxdepth)
        o_cell = self.discretizer.transform(o.reshape(1, -1)).astype(int)[0]
        gathered_elements = self.query_cell_frequency(o_cell) + self.rnd_generator.laplace(scale=laplace_noise_sd)
        dist = 0
        i = 0
        while gathered_elements < k:
            i = i + 1
            try:
                next_cell, dist_to_cell = trav.next_cell()
            except TypeError:
                # gathered elements not reached. add penalty
                return self.number_of_dimensions

            next_freq = self.query_cell_frequency(next_cell) + self.rnd_generator.laplace(scale=laplace_noise_sd)
            gathered_elements = gathered_elements + next_freq

            if (i % 2**16) == 0:
                print("el = {el} for iteration, expecting {k}".format(k=k,el=gathered_elements))

            if weighted:
                dist = dist + dist_to_cell * next_freq
            else:
                dist = dist_to_cell
        return dist

    def batch_query(self, o_array , k, weighted=True):
        """
            Similar as query, however takes an o_array and only considers populated cells
            implementation wise. Furthermore, it will extend beyond asjacent cells.
            Returns the l1 lattice neighbor distance.

            Parameters
            ----------
            o : ndarray of points
            A point in the data space, for which we want the lattice distance to neighbor cells depending on k

            k : int
            The number of cell frequencies desired (lower bound when noisy)


            weighted=True : boolean
            Whether to return the distances summed and weighted by number of elements per cell
             or only return final distance to cell which exceeded threshold k.

            Returns
            -------
            dist : int
                L1 distance in unit according to cell width.
        """
        o_array_centroids = self.discretizer.inverse_transform(self.discretizer.transform(o_array))
        distances, _ = self.bin_tree.query(o_array_centroids, k=k)
        if weighted:
            return np.sum(distances, axis=1)
        else:
            return distances[:, -1]


class BinnedKnnTraversal:
    def __init__(self, binned_knn, o, max_depth=8):
        assert ((o >= 0) & (o <= 1)).all()

        self.binned_knn = binned_knn
        self.o = o
        self.max_depth = max_depth

        self.o_cell = self.binned_knn.discretizer.transform(o.reshape(1, -1)).astype(int)[
            0]  # TODO remove reshape and make vector compatible
        self.borders = self.binned_knn.get_point_cell_borders(self.o_cell)

        self.MAX_DIM = 32 # limit due to bit representation

        self.corner_order = None
        self.current_order_dist = None
        self.current_depth = 1
        self.setup_corner_order(self.current_depth)
        self.current_corner_masks = iter(())
        self.next_corner_idx = None

    def next_corner_cell(self):
        """
            Returns the next adjacent cell in the corners of the cell corresponding to self.o.

            Returns
            -------
            next_cell : ndarray
                An ordinal encoding of the next corner cell
        """
        while True:
            cell = self.o_cell.copy()
            try:
                corner_mask = next(self.current_corner_masks)
            except StopIteration:
                self.next_corner_idx = next(self.corner_order)
                # the corner_order only considers the order w/r to distances IN o_cell
                # so that we need to consider additionally the depth combinations from here
                self.current_corner_masks = util.ordinal_masks_for_combination_idx(self.next_corner_idx,
                                                                                   self.binned_knn.number_of_dimensions,
                                                                                   self.current_depth)
                corner_mask = next(self.current_corner_masks)

            next_cell = cell + corner_mask
            if not self.binned_knn.is_cell_ordinal_in_range(next_cell):
                continue
            dist = self.current_order_dist[self.next_corner_idx]
            return next_cell, dist

    def setup_corner_order(self, depth):
        corner_order, self.current_order_dist = util.get_cell_corner_order(self.o, self.borders,
                                                                           depth)
        iterator = itertools.takewhile(lambda x: (not math.isnan(self.current_order_dist[x])), corner_order)
        self.corner_order = iterator

    def next_cell(self):
        """
            Returns the next adjacent cell

            Returns
            -------
            next_cell : ndarray
                An ordinal encoding of the next cell
            dist : int
                The l1 distance to this cell from the centroid of cell in which o is located.
                Distance is given in bin_width units, i.e. for width 1 horizontal cells are 1 unit apart
                and corner cells are d units apart
        """
        while self.current_depth <= self.max_depth:
            try:
                next_cell, dist_to_cell = self.next_corner_cell()
                return next_cell, dist_to_cell
            except StopIteration:
                self.current_depth = self.current_depth + 1
                self.setup_corner_order(self.current_depth)
        warnings.warn("Max depth reached")
        return None

