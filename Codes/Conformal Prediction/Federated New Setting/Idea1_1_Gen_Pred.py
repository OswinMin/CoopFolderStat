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

# 生成不同 X 来自 N(loc, 3)
# 对应 X 的 Y = X**2 + N(0, (ep*|X|)**2)

ep = 0.5 # scale(sigma)
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
        X[:, i] = np.random.normal(loc[i], 3, n)
        Y[:, i] = f(X[:, i])
    return X, Y

def fun(X):
    """
    X -> Y
    :param X: ndarray of length [n]
    :return: X^2 + ep*|X|
    """
    return X**2 + (np.abs(X))*ep*np.random.normal(0, 1, len(X))

def trueConf(X):
    return np.vstack([X**2+stats.norm.ppf(0.05)*ep*(np.abs(X)),
                      X**2+stats.norm.ppf(0.95)*ep*(np.abs(X))])

def genLoc(num, loc):
    """
    generate loc matrix
    :param num: number of samples each agent, n
    :param loc: location of each agent, length k
    :return: n*k np.ndarray, n copies of loc
    """
    loc_ = np.ones((num, len(loc)))
    return loc_*loc

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


def SaveData(X, Y, loc, st):
    num = X.shape[0]
    np.save(f"data/pureCovariateShift/{st}_X.npy", X)
    np.save(f"data/pureCovariateShift/{st}_Y.npy", Y)
    np.save(f"data/pureCovariateShift/{st}_loc.npy", loc)
    with open(f"data/pureCovariateShift/{st}_meta.txt", 'w', encoding='utf-8') as f:
        f.write("生成不同 X 来自 N(loc, 3)\n")
        f.write("对应 X 的 Y = X**2 + N(0, (ep*|X|)**2)\n")
        f.write(f"loc:{[round(i,2) for i in loc]}\n")
        f.write(f"Agent num:{len(loc)}\n")
        f.write(f"每个Agent拥有样本数:{num}")

def LoadData(st):
    X = np.load(f"data/pureCovariateShift/{st}_X.npy")
    Y = np.load(f"data/pureCovariateShift/{st}_Y.npy")
    loc = np.load(f"data/pureCovariateShift/{st}_loc.npy")
    return X, Y, loc

if __name__ == '__main__':
    pass
    # X, Y = Xshift(500, [0, 0.1, 0.2, -0.05, 0.6, -1], fun)
    # predictor = Predictor()
    # predictor.train(X.reshape(-1,1), Y.reshape(-1,1))