import numpy as np

class ScoringMetrics(object):
    """Scoring metrics

    Args:
        object (class): Module for various scoring metrics
    """
    def __init__(self, pred, target):
        """Intialise variable

        Args:
            pred (numpy): Predicted variable
            target (numpy): Target variable
        """
        self.pred = pred
        self.target = target
        self.n_clusters = len(np.unique(self.pred))

    def accuracy_score(self,):
        """Performs classification scoring
        """
        # First match predicted labels cluster labels to target
        labels=np.zeros_like(self.pred)
        for cluster in range(self.n_clusters):
            mask = (self.pred==cluster)
            labels[mask]=np.bincount(self.target[mask]).argmax(axis=0)

        return (sum(labels == self.target))/self.target.shape[0]