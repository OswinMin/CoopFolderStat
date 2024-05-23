import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def Xshift(n, loc, f):
    """
    X has shift but Y|X remains same
    :param n: each agent has n samples
    :param loc: ndarray length [k] each agent has distribution from N(loc[i],1)
    :param f: X -> Y with randomness
    :return: ndarray n*k, ndarray n*k, X and Y
    """
    X = np.zeros((n, len(loc)))
    Y = np.zeros((n, len(loc)))
    for i in range(len(loc)):
        X[:, i] = np.random.normal(loc[i], 1, n)
        Y[:, i] = f(X[:, i])
    return X, Y

def fun(X):
    """
    X -> Y
    :param X: ndarray of length [n]
    :return: X^2 + ep*|X|
    """
    return X*2 + np.abs(X)*np.random.normal(0, 0.03, len(X))

class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()
        self.l1 = nn.Linear(1, 8)
        self.l2 = nn.Linear(8, 4)
        self.l3 = nn.Linear(4, 1)
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
    X, Y = Xshift(500, [0, 0.1, 0.2, -0.05, 0.6, -1], fun)
    predictor = Predictor()
    predictor.train(X.reshape(-1,1), Y.reshape(-1,1))