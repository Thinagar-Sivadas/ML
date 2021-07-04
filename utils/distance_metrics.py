import numpy as np

class DistanceMetrics(object):
    """Distance metrics

    Args:
        object (class): Module for various distance metrics
    """

    def __init__(self, var1, var2):
        """Intialise variable

        Args:
            var1 (numpy): Variable 1
            var2 (numpy): Variable 2
        """
        var1 = var1.astype('float64')
        var2 = var2.astype('float64')
        self.var1 = var1
        self.var2 = var2

    def eucliden_distance(self):
        """Performs euclidean distance
        https://medium.com/swlh/euclidean-distance-matrix-4c3e1378d87f

        Returns:
            numpy array: var1.shape[0], var2.shape[0]
        """
        A = np.sum(self.var1**2,axis=1).reshape(-1, 1)
        B = np.sum(self.var2**2,axis=1).reshape(1, -1)
        C = -2 * (np.dot(self.var1, self.var2.T))
        return np.sqrt(np.abs(A+B+C))