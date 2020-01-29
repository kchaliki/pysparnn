import random as _random
import numpy as _np

import collections as _collections

from abc import ABC, abstractmethod

from sklearn.cluster import DBSCAN


def _k_best(tuple_list, k):
    """For a list of tuples [(distance, value), ...] - Get the k-best tuples by
    distance.
    Args:
        tuple_list: List of tuples. (distance, value)
        k: Number of tuples to return.
    """
    tuple_lst = sorted(tuple_list, key=lambda x: x[0],
                       reverse=False)[:k]

    return tuple_lst


class ClusterSelector(ABC):

    @abstractmethod
    def select_clusters(self, features):
        pass


class DefaultClusterSelector(ClusterSelector):
    """
    Default cluster selector, picks sqrt(num_records) random points (at most 1000)
    and allocates points to their nearest category. This can often end up splitting
    similar points into multiple paths of the tree
    """

    def __init__(self, distance_type):
        self._distance_type = distance_type

    def select_clusters(self, features):
        # number of points to cluster
        num_records = features.shape[0]

        matrix_size = max(int(_np.sqrt(num_records)), 1000)

        # set num_clusters = min(max(sqrt(num_records), 1000), num_records))
        clusters_size = min(matrix_size, num_records)

        # make list [0, 1, ..., num_records-1]
        records_index = list(_np.arange(features.shape[0]))
        # randomly choose num_clusters records as the cluster roots
        # this randomizes both selection and order of features in the selection
        clusters_selection = _random.sample(records_index, clusters_size)
        clusters_selection = features[clusters_selection]

        # create structure to store clusters
        item_to_clusters = _collections.defaultdict(list)

        # create a distance_type object containing the cluster roots
        # labeling them as 0 to N-1 in their current (random) order
        root = self._distance_type(clusters_selection,
                             list(_np.arange(clusters_selection.shape[0])))

        # remove duplicate cluster roots
        root.remove_near_duplicates()
        # initialize distance type object with the remaining cluster roots
        root = self._distance_type(root.matrix,
                             list(_np.arange(root.matrix.shape[0])))

        rng_step = matrix_size
        # walk features in steps of matrix_size = max(sqrt(num_records), 1000)
        for rng in range(0, features.shape[0], rng_step):
            # don't exceed the array length on the last step
            max_rng = min(rng + rng_step, features.shape[0])
            records_rng = features[rng:max_rng]
            # find the nearest cluster root for each feature in the step
            for i, clstrs in enumerate(root.nearest_search(records_rng)):
                _random.shuffle(clstrs)
                for _, cluster in _k_best(clstrs, k=1):
                    # add each feature to its nearest cluster, here the cluster label
                    # is the label assigned to the root feature after it had been selected at random
                    item_to_clusters[cluster].append(i + rng)

        # row index in clusters_selection maps to key in item_to_clusters
        # but the values in item_to_clusters are row indices of the original features matrix
        return clusters_selection, item_to_clusters


class DbscanClusterSelector(ClusterSelector):
    """
    Dbscan based cluster selector, picks sqrt(num_records) random points (at most 1000)
    and then forms groups inside the random selection, before allocating other features
    to the groups
    """

    def __init__(self, distance_type):
        self._distance_type = distance_type
        self._eps = 0.4

    def select_clusters(self, features):
        # number of points to cluster
        num_records = features.shape[0]

        matrix_size = max(int(_np.sqrt(num_records)), 1000)

        # set num_clusters = min(max(sqrt(num_records), 1000), num_records))
        clusters_size = min(matrix_size, num_records)

        # make list [0, 1, ..., num_records-1]
        records_index = list(_np.arange(features.shape[0]))
        # randomly choose num_clusters records as the cluster roots
        # this randomizes both selection and order of features in the selection
        random_clusters_selection = _random.sample(records_index, clusters_size)
        random_clusters_selection = features[random_clusters_selection]

        # now cluster the cluster roots themselves to avoid
        # randomly separating neighbours, this probably means fewer clusters per level
        # TODO might want to propagate the distance type to the clustering
        db_scan_clustering = DBSCAN(eps=self._eps, min_samples=2).fit(random_clusters_selection)

        # get all the individual points from the cluster
        unique_indices = _np.where(db_scan_clustering.labels_ == -1)[0]
        # and the first item from each cluster
        _, cluster_start_indices = _np.unique(db_scan_clustering.labels_, return_index=True)
        # merge and uniquefy, the result is sorted
        all_indices = _np.concatenate((unique_indices, cluster_start_indices))
        all_indices_unique = _np.unique(all_indices)

        # create a matrix where rows are the first item in each dbscan cluster
        # set that as cluster selection and then allocate features to cluster
        clusters_selection = random_clusters_selection[all_indices_unique]

        # create structure to store clusters
        item_to_clusters = _collections.defaultdict(list)

        # create a distance_type object containing the cluster root
        root = self._distance_type(clusters_selection,
                                   list(_np.arange(clusters_selection.shape[0])))

        rng_step = matrix_size
        # walk features in steps of matrix_size = max(sqrt(num_records), 1000)
        for rng in range(0, features.shape[0], rng_step):
            max_rng = min(rng + rng_step, features.shape[0])
            records_rng = features[rng:max_rng]
            # find the nearest cluster root for each feature in the step
            for i, clstrs in enumerate(root.nearest_search(records_rng)):
                # this is slow, disable until proven useful
                # _random.shuffle(clstrs)
                for _, cluster in _k_best(clstrs, k=1):
                    # add each feature to its nearest cluster
                    item_to_clusters[cluster].append(i + rng)

        # row index in clusters_selection maps to key in item_to_clusters
        # but the values in item_to_clusters are row indices of the original features matrix
        return clusters_selection, item_to_clusters
