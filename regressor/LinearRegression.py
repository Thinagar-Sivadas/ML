import numpy as np

class LinearRegression(object):
    """Linear Regression optimisation using gradient descent or closed form solution
    """

    def __init__(self, lr=1e-3, max_iter=500):
        """Intialise variables

        Args:
            lr ([type], optional): [description]. Defaults to 1e-3.
            max_iter (int, optional): [description]. Defaults to 300.
        """

        self.lr = lr
        self.max_iter = max_iter
        self.weights = None
        self.bias = None

    def _init_params(self):
        self.weights = np.zeros(shape=(self.train_data.shape[1], 1))
        self.bias = np.zeros(shape=(1, 1))

    def train(self, train_data, train_target):

        self.train_data = train_data.astype('float64')
        self.train_target = train_target.astype('float64').reshape(-1, 1)

        if self.weights is None and self.bias is None:
            self._init_params()

        for _ in range(self.max_iter):
            y_hat = self.predict(data=train_data)

            delta_weights = (-2/self.train_data.shape[0]) * np.dot(self.train_data.T, (self.train_target-y_hat))
            delta_bias = (-2/self.train_data.shape[0]) * np.sum(self.train_target-y_hat, keepdims=True)

            self.weights -= (self.lr*delta_weights)
            self.bias -= (self.lr*delta_bias)

    def predict(self, data):
        return np.dot(data, self.weights) + self.bias