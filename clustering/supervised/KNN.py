import numpy as np
from utils.distance_metrics import DistanceMetrics
from utils.scoring_metrics import ScoringMetrics
from utils.plot_knn import PlotKNN

class KNN(DistanceMetrics, PlotKNN, ScoringMetrics):
    """KNN

    Args:
        DistanceMetrics (obj): Performs distance metrics calculation
        PlotKNN (obj): Performs plot
        ScoringMetrics (obj): Performs scoring metrics calculation
    """

    def __init__(self, n_neighbors):
        """Intialise variables

        Args:
            n_neighbors (int): Number of neighbours
        """
        self.n_neighbors = n_neighbors
        self.train_data = None
        self.train_labels = None
        self.n_features = None
        self.n_clusters = None

    def train(self, train_data, train_labels):
        """Training of knn

        Args:
            train_data (numpy): Training data
            train_labels (numpy): Train labels
        """
        print('------Training------')
        self.train_data = train_data.astype('float64')
        self.train_labels = train_labels.astype('int64')
        self.n_features = self.train_data.shape[1]
        self.n_clusters = len(np.unique(self.train_labels))

    def predict(self, data):
        """Predict datapoints cluster labels

        Args:
            data (numpy): Data to predict

        Returns:
            numpy: Cluster labels
        """
        DistanceMetrics.__init__(self,
                                 var1=data,
                                 var2=self.train_data)
        k_indices = np.argsort(self.eucliden_distance(), axis=1)[:, 0:self.n_neighbors]
        return np.array([np.bincount(row).argmax(axis=0) for row in self.train_labels[k_indices]])

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

    def plot_training(self):
        """Plot training
        """

        PlotKNN.__init__(self)

        if self.n_features == 1:
            self.plot_training_1d(
                train_data=self.train_data, n_clusters=self.n_clusters,
                train_labels=self.train_labels)

        elif self.n_features == 2:
            self.plot_training_2d(
                train_data=self.train_data, n_clusters=self.n_clusters,
                train_labels=self.train_labels)

        elif self.n_features == 3:
            self.plot_training_3d(
                train_data=self.train_data, n_clusters=self.n_clusters,
                train_labels=self.train_labels)

        else:
            return f"Cannot visualise for {self.n_features} dimensional data"

    def plot_testing(self, test_data):
        """Plot testing

        Args:
            test_data (numpy): Test data
        """

        PlotKNN.__init__(self)

        if self.n_features == 1:
            self.plot_testing_1d(
                train_data=self.train_data, test_data=test_data,
                n_clusters=self.n_clusters, train_labels=self.train_labels,
                test_labels=self.predict(test_data)
                )

        elif self.n_features == 2:
            self.plot_testing_2d(
                    train_data=self.train_data, test_data=test_data,
                    n_clusters=self.n_clusters, train_labels=self.train_labels,
                    test_labels=self.predict(test_data)
                    )

        elif self.n_features == 3:
            self.plot_testing_3d(
                train_data=self.train_data, test_data=test_data,
                n_clusters=self.n_clusters, train_labels=self.train_labels,
                test_labels=self.predict(test_data)
                )

        else:
            return f"Cannot visualise for {self.n_features} dimensional data"