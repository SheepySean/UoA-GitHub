from sklearn.cluster import DBSCAN
from ML.clusterizer import Clusterizer


class DBSCANClusterizer(Clusterizer):
    def compute(self, eps=0.3, min_samples=10):
        dbscan = DBSCAN(eps=0.3, min_samples=min_samples, n_jobs=self._jobs, algorithm='ball_tree').fit(
            self._preprocessed_data.toarray())
        self._labels = dbscan.labels_

        return dbscan
