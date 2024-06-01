import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import scipy.stats as stats
"""
函数名以 _1 结尾是第一种情形的实验，只有协变量迁移，选取 k 个 agent
每个 agent 选取某个 location(loc) 和标准差来自 [0.5,0.8,1,1.5,2](sigma)
每个 agent 生成 n 个样本 X 来自 N(loc,sigma^2)
Y|X = X^2 + |ep*X|*N(0,1)
X, Y: n*k
para: 2*k       第一行是 location，第二行是 sigma
"""

ep = 0.5 # scale(sigma)

def Xshift_1(n, loc, f):
    """
    X has shift but Y|X remains same
    :param n: each agent has n samples
    :param loc: ndarray length [k] each agent has distribution from N(loc[i],1)
    :param f: X -> Y with randomness
    :return: ndarray n*k, ndarray n*k, X and Y,
            parameter 2*k(location and sigma)
    """
    X = np.zeros((n, len(loc)))
    Y = np.zeros((n, len(loc)))
    sigma = np.random.choice([0.5,0.8,1,1.5,2], len(loc))
    for i in range(len(loc)):
        X[:, i] = np.random.normal(loc[i], sigma[i], n)
        Y[:, i] = f(X[:, i])
    return X, Y, np.vstack([loc, sigma])

def fun_1(X):
    """
    X -> Y
    :param X: ndarray of length [n]
    :return: X^2 + ep*|X|*N(0,1)
    """
    return X**2 + (np.abs(X))*ep*np.random.normal(0, 1, len(X))

def trueConf1(X):
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

def SaveData_1(X, Y, para, st):
    num = X.shape[0]
    np.save(f"data/pureCovariateShift/{st}_X.npy", X)
    np.save(f"data/pureCovariateShift/{st}_Y.npy", Y)
    np.save(f"data/pureCovariateShift/{st}_para.npy", para)
    with open(f"data/pureCovariateShift/{st}_meta.txt", 'w', encoding='utf-8') as f:
        f.write("生成不同 X 来自 N(loc, sigma)\n")
        f.write("对应 X 的 Y = X**2 + N(0, (ep*|X|)**2)\n")
        f.write(f"  loc: "+','.join([f'{i:5}' for i in para[0,:]])+'\n')
        f.write(f"sigma: "+','.join([f'{i:5}' for i in para[1,:]])+'\n')
        f.write(f"Agent个数:{para.shape[1]}\n")
        f.write(f"每个Agent拥有样本数:{num}")

def LoadData1(st):
    X = np.load(f"data/pureCovariateShift/{st}_X.npy")
    Y = np.load(f"data/pureCovariateShift/{st}_Y.npy")
    para = np.load(f"data/pureCovariateShift/{st}_para.npy")
    return X, Y, para

if __name__ == '__main__':
    X, Y, para = Xshift_1(500, [-10,-5,0,5,10], fun_1)
    SaveData_1(X, Y, para, "FULL")
    SaveData_1(X[:, 2:], Y[:, 2:], para[:, 2:], "PART")
    X, Y, para = Xshift_1(50, [-10, -10, -5, -5, 0, 5, 5, 10, 10], fun_1)
    SaveData_1(X, Y, para, "SPARSEFULL")
    SaveData_1(X[:, 4:], Y[:, 4:], para[:, 4:], "SPARSEPART")
    X, Y, para = Xshift_1(300, [0] + [12] * 29, fun_1)
    SaveData_1(X, Y, para, "FARFULL")
    SaveData_1(X[:, [0]], Y[:, [0]], para[:, [0]], "FARPART")
    X, Y, para = Xshift_1(40, [0] + [12] * 39, fun_1)
    SaveData_1(X, Y, para, "FARSPARSEFULL")
    SaveData_1(X[:, [0]], Y[:, [0]], para[:, [0]], "FARSPARSEPART")