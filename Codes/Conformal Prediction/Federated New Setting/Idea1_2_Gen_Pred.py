import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import scipy.stats as stats
"""
！！！！！！！注意！！！！！！！！！
        本文档代码已废弃
           留存作备份
          请勿运行代码
！！！！！！！注意！！！！！！！！！
"""
ep = 0.5 # scale(sigma)
def Xshift(n, loc, f):
    """
    X has shift but Y|X remains same
    :param n: each agent has n samples
    :param loc: ndarray length [k] each agent has distribution from N(loc[i],1)
    :param f: X -> Y with randomness
    :return: ndarray n*k, ndarray n*k,, ndarray k, X and Y and theta
    """
    X = np.zeros((n, len(loc)))
    Y = np.zeros((n, len(loc)))
    theta = np.random.uniform(1, 10, len(loc))
    for i in range(len(loc)):
        X[:, i] = np.random.normal(loc[i], 3, n)
        Y[:, i] = f(X[:, i], theta[i])
    return X, Y, theta

def fun(X, theta):
    """
    X -> Y
    :param X: ndarray of length [n]
    :param theta: parameter
    :return: X^2 + ep*(|X|+theta) N(0,1)
    """
    return X**2 + (np.abs(X)+theta)*ep*np.random.normal(0, 1, len(X))

def trueConf(X, theta, alpha=0.05):
    """
    Return confidence bound for each X_i^k
    :param X: data matrix n*k
    :param theta: data parameter k
    :param alpha: threshold
    :return: lowerbound, upperbound (shape like X)
    """
    lowerbound = X**2 + stats.norm.ppf(0.05)*ep*(np.abs(X)+theta)
    upperbound = X**2 + stats.norm.ppf(0.95)*ep*(np.abs(X)+theta)
    return lowerbound, upperbound

class Predictor(nn.Module):
    def __init__(self, hidden):
        """
        :param hidden: list of length 2
        """
        super(Predictor, self).__init__()
        self.l1 = nn.Linear(1, hidden[0])
        self.l2 = nn.Linear(hidden[0], hidden[1])
        self.l3 = nn.Linear(hidden[1], 1)
        self.R = nn.ReLU()

    def forward(self, x):
        x = self.R(self.l1(x))
        x = self.R(self.l2(x))
        return self.l3(x)

    def train(self, X, Y, batch_size=32, epochs=100, learning_rate=0.01):
        """
        Use X, Y train a simple predictor
        :param X:
        :param Y:
        :param batch_size:
        :param epochs:
        :param learning_rate:
        :return: No return
        """
        X_tensor = torch.tensor(X, dtype=torch.float32)
        Y_tensor = torch.tensor(Y, dtype=torch.float32)

        # Create a dataset and data loader
        dataset = TensorDataset(X_tensor, Y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(epochs):
            for inputs, targets in dataloader:
                # Forward pass
                outputs = self(inputs)
                loss = criterion(outputs, targets)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Print loss for every epoch
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

if __name__ == '__main__':
    X, Y, theta = Xshift(1000, [0, 1], fun)
    lowerbound, upperbound = trueConf(X, theta)
    print(np.sum(Y<lowerbound), np.sum(Y>upperbound))