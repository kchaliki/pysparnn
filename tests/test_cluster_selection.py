import numpy as np
from scipy.sparse import csr_matrix

from pysparnn.cluster_selection import DefaultClusterSelector, DbscanClusterSelector
from pysparnn.matrix_distance import CosineDistance, SlowEuclideanDistance


class TestDefaultClusterSelector(object):
    def test_single_item_groups(self):
        sel = DefaultClusterSelector(CosineDistance)

        features_dense = np.identity(1000)
        features = csr_matrix(features_dense)
        cs, i2c = sel.select_clusters(features)

        assert cs.shape[0] == 1000
        assert all(len(cl) == 1 for cl in i2c.values())

    def test_non_single_item_groups(self):
        sel = DefaultClusterSelector(SlowEuclideanDistance)

        features_dense = np.identity(1001)
        # features = csr_matrix(features_dense)
        cs, i2c = sel.select_clusters(features_dense)

        assert cs.shape[0] == 1000
        assert all(len(cl) >= 1 for cl in i2c.values())
        assert sum(len(cl) for cl in i2c.values()) == 1001

        non_single_groups = list(cl for cl in i2c.values() if len(cl) == 2)
        assert len(non_single_groups) == 1


class TestDbscanClusterSelector(object):

    def test_one(self):
        sel = DbscanClusterSelector(CosineDistance)

        features_dense = np.identity(100)
        similar = [[0.9] + [0] * 99]
        features_dense = np.append(features_dense, similar, axis=0)
        assert features_dense.shape[0] == 101

        features = csr_matrix(features_dense)
        cs, i2c = sel.select_clusters(features)

        assert cs.shape[0] == 100
        assert all(len(cl) >= 1 for cl in i2c.values())
        assert sum(len(cl) for cl in i2c.values()) == 101

        non_single_groups = list(cl for cl in i2c.values() if len(cl) == 2)
        assert len(non_single_groups) == 1
