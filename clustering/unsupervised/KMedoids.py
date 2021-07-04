import numpy as np
from utils.distance_metrics import DistanceMetrics
from utils.scoring_metrics import ScoringMetrics
from utils.plot_kmedoids import PlotKMedoids

class KMedoids(DistanceMetrics, PlotKMedoids, ScoringMetrics):
    """KMedoids
    https://towardsdatascience.com/understanding-k-means-k-means-and-k-medoids-clustering-algorithms-ad9c9fbf47ca

    Args:
        DistanceMetrics (obj): Performs distance metrics calculation
        PlotKMedoids (obj): Performs plot
        ScoringMetrics (obj): Performs scoring metrics calculation
    """

    def __init__(self, n_clusters, init='k-medoids++', max_iter=300, tol=1e-4, centroids=None, n_init=10):
        """Intialise variables

        Args:
            n_clusters (int): Number of clusters
            init (str, optional): Defaults to 'random'.
            max_iter (int, optional): Defaults to 300.
            tol (float, optional): Defaults to 1e-4.
            centroids (numpy, optional): Defaults to None.
        """

        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = centroids
        self.train_data = None
        self.train_labels = None
        self.n_features = None
        self.hist_centroids = []
        self.hist_labels = []
        self.n_init = n_init
        self.inertia = None

    def _init_centroids(self):
        """Intialise centroids if not specified
        https://www.geeksforgeeks.org/ml-k-means-algorithm/
        """

        if self.init == 'random':
            idx = np.random.choice(self.train_data.shape[0], self.n_clusters, replace=False)
            self.centroids = self.train_data[idx, :]
            print("Centroids intialised randomly")

        elif self.init == 'k-medoids++':
            centroids = []
            data = self.train_data
            idx = np.random.choice(data.shape[0], 1, replace=False)
            centroids.append(data[idx, :])
            data = np.delete(data, idx, axis=0)

            for _ in range(1, self.n_clusters):
                DistanceMetrics.__init__(self,
                                 var1=data,
                                 var2=np.vstack(centroids))
                idx = [np.argmax(self.eucliden_distance().min(axis=1))]
                centroids.append(data[idx, :])
                data = np.delete(data, idx, axis=0)

            self.centroids = np.vstack(centroids)
            print("Centroids intialised using k-medoids++")

    def _tol_criteria(self):
        """Tolerance criterion check

        Returns:
            float: Frobenius(Euclidean) norm of consecutive centroids difference
        """

        consec_centroids_diff = self.hist_centroids[-1] - self.hist_centroids[-2]
        return np.sqrt((consec_centroids_diff**2).sum()) <= self.tol

    def _cal_inertia(self,):
        """Calculate inertia
        """

        inertia = 0
        for n_cluster in range(self.n_clusters):
            mask = (self.train_labels == n_cluster)
            DistanceMetrics.__init__(self,
                                     var1=self.train_data[mask],
                                     var2=self.centroids[[n_cluster]])
            inertia += (self.eucliden_distance()**2).sum()
        self.inertia = inertia

    def _train(self):
        """Training of kmedoids
        """

        if self.centroids is None:
            self._init_centroids()
        self.hist_centroids.append(self.centroids)

        for _ in range(self.max_iter):
            labels = self.predict(data=self.train_data)
            self.hist_labels.append(labels)
            # Update centroids
            centroids = []
            for n_cluster in range(self.n_clusters):
                var = self.train_data[labels == n_cluster]
                DistanceMetrics.__init__(self,
                                         var1=var,
                                         var2=var)
                centroids.append(var[np.argmin(self.eucliden_distance().sum(axis=1))])
            self.centroids = np.array(centroids)
            self.hist_centroids.append(self.centroids)
            if self._tol_criteria() == True:
                print(f'Tolerance criterion reached at iteration {len(self.hist_centroids)-1}')
                break

        self.train_labels = self.predict(data=self.train_data)
        self.hist_labels.append(self.train_labels)
        self._cal_inertia()

    def train(self, train_data):
        """Training based on n_init. Determine best init based on inertia

        Args:
            train_data (numpy): Training data
        """

        print('------Training------', end='\n\n')
        self.train_data = train_data.astype('float64')
        self.n_features = self.train_data.shape[1]
        n_init_centroids = []
        n_init_hist_centroids = []
        n_init_hist_labels = []
        n_init_hist_train_labels = []
        n_init_inertia = []

        for ind in range(self.n_init):
            print(f"Init {ind + 1}")
            self._train()
            print()
            # Save init params
            n_init_centroids.append(self.centroids)
            n_init_hist_centroids.append(self.hist_centroids)
            n_init_hist_labels.append(self.hist_labels)
            n_init_hist_train_labels.append(self.train_labels)
            n_init_inertia.append(self.inertia)
            # Clear params for next init
            (self.centroids, self.hist_centroids,
            self.hist_labels, self.train_labels, self.inertia) = (None, [], [], None, None)

        index_inertia_min = np.argmin(n_init_inertia)
        print(f"Init {index_inertia_min+1} chosen. " +
              f"It has the lowest inertia of {n_init_inertia[index_inertia_min]:.3f}")
        (self.centroids, self.hist_centroids,
         self.hist_labels, self.train_labels, self.inertia) = (n_init_centroids[index_inertia_min],
                                                               n_init_hist_centroids[index_inertia_min],
                                                               n_init_hist_labels[index_inertia_min],
                                                               n_init_hist_train_labels[index_inertia_min],
                                                               n_init_inertia[index_inertia_min])

    def predict(self, data):
        """Predict datapoints cluster labels

        Args:
            data (numpy): Data to predict

        Returns:
            numpy: Cluster labels
        """

        DistanceMetrics.__init__(
            self,
            var1=data,
            var2=self.centroids
            )
        return np.argmin(self.eucliden_distance(), axis=1)

    def evaluate_score(self, pred_labels, target_labels):
        """Scoring

        Args:
            pred_labels (numpy): Predicted labels
            target_labels (numpy): Target labels

        Returns:
            int: Score
        """

        ScoringMetrics.__init__(self, pred=pred_labels, target=target_labels)
        return self.accuracy_score()

    def plot_train(self):
        """Plot train
        """

        PlotKMedoids.__init__(self)

        if self.n_features == 1:
            self.plot_training_1d(
                train_data=self.train_data, n_clusters=self.n_clusters,
                hist_labels=self.hist_labels, hist_centroids=self.hist_centroids
                )

        elif self.n_features == 2:
            self.plot_training_2d(
                train_data=self.train_data, n_clusters=self.n_clusters,
                hist_labels=self.hist_labels, hist_centroids=self.hist_centroids
                )

        elif self.n_features == 3:
            self.plot_training_3d(
                train_data=self.train_data, n_clusters=self.n_clusters,
                hist_labels=self.hist_labels, hist_centroids=self.hist_centroids
                )

        else:
            return f"Cannot visualise for {self.n_features} dimensional data"

    def plot_test(self, test_data):
        """Plot test

        Args:
            test_data (numpy): Test data
        """

        PlotKMedoids.__init__(self)

        if self.n_features == 1:
            self.plot_testing_1d(
                train_data=self.train_data, test_data=test_data,
                n_clusters=self.n_clusters, train_labels=self.train_labels,
                test_labels=self.predict(test_data), centroids=self.centroids
                )

        elif self.n_features == 2:
            self.plot_testing_2d(
                train_data=self.train_data, test_data=test_data,
                n_clusters=self.n_clusters, train_labels=self.train_labels,
                test_labels=self.predict(test_data), centroids=self.centroids
                )

        elif self.n_features == 3:
            self.plot_testing_3d(
                train_data=self.train_data, test_data=test_data,
                n_clusters=self.n_clusters, train_labels=self.train_labels,
                test_labels=self.predict(test_data), centroids=self.centroids
                )

        else:
            return f"Cannot visualise for {self.n_features} dimensional data"