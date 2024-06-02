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
    X has shift but Y|X remains same, sigma comes from [0.5,0.8,1,1.5,2]
    :param n: each agent has n samples
    :param loc: ndarray length [k] each agent has distribution from N(loc[i],sigma[i])
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

def Xshift_2(n, loc, f):
    """
    X has shift, Y|X also has random shift, sigma comes from [0.5,0.8,1,1.5,2]
    theta comes from [0,2,3,4,5,6,7,8]
    :param n: each agent has n samples
    :param loc: ndarray length [k] each agent has distribution from N(loc[i],sigma[i])
    :param f: X -> Y with randomness
    :return: ndarray n*k, ndarray n*k, X and Y,
            parameter 3*k(location,sigma of X, theta of Y's deviation)
    """
    X = np.zeros((n, len(loc)))
    Y = np.zeros((n, len(loc)))
    sigma = np.random.choice([0.5,0.8,1,1.5,2], len(loc))
    theta = np.random.choice(np.arange(3,9), len(loc))
    theta[np.array(loc)==0] = 0
    for i in range(len(loc)):
        X[:, i] = np.random.normal(loc[i], sigma[i], n)
        Y[:, i] = f(X[:, i], theta[i])
    return X, Y, np.vstack([loc, sigma, theta])

def fun_2(X, theta):
    """
    X -> Y
    :param X: ndarray of length [n]
    :param theta: float > 0, some parameter
    :return: X^2 + ep*|X|*N(0,1)
    """
    return X**2 + (np.abs(X)+theta)*ep*np.random.normal(0, 1, len(X))

def trueConf2(X, theta, cov=0.9):
    """
    Return confidence bound for each X comes from agent with parameter theta
    :param X: ndarray of length [n]
    :param theta: ndarray of length [n]
    :param cov: coverage
    :return: ndarray [lowerbound, upperbound] 2*n
    """
    lowerbound = X**2 + stats.norm.ppf((1-cov)/2)*ep*(np.abs(X)+theta)
    upperbound = X**2 + stats.norm.ppf(1-(1-cov)/2)*ep*(np.abs(X)+theta)
    return np.vstack([lowerbound, upperbound])

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
    """
    保存数据
    :param X: X n*k
    :param Y: Y n*k
    :param para: loc and sigma 2*k
    :param st: 文件名
    :return: None
    """
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

def SaveData_2(X, Y, para, st):
    """
    保存数据
    :param X: X n*k
    :param Y: Y n*k
    :param para: loc and sigma and theta, 3*k
    :param st: 文件名
    :return: None
    """
    num = X.shape[0]
    np.save(f"data/CovariateRandomShift/{st}_X.npy", X)
    np.save(f"data/CovariateRandomShift/{st}_Y.npy", Y)
    np.save(f"data/CovariateRandomShift/{st}_para.npy", para)
    with open(f"data/CovariateRandomShift/{st}_meta.txt", 'w', encoding='utf-8') as f:
        f.write("生成不同 X 来自 N(loc, sigma)\n")
        f.write("对应 X 的 Y = X**2 + N(0, (ep*|X|)**2)\n")
        f.write(f"  loc: "+','.join([f'{i:5}' if not i.is_integer()
                                     else f'{int(i):5}'
                                     for i in para[0,:]])+'\n')
        f.write(f"sigma: "+','.join([f'{i:5}' if not i.is_integer()
                                     else f'{int(i):5}'
                                     for i in para[1,:]])+'\n')
        f.write(f"theta: " + ','.join([f'{i:5}' if not i.is_integer()
                                       else f'{int(i):5}'
                                       for i in para[2, :]]) + '\n')
        f.write(f"Agent个数:{para.shape[1]}\n")
        f.write(f"每个Agent拥有样本数:{num}")

def LoadData2(st):
    X = np.load(f"data/CovariateRandomShift/{st}_X.npy")
    Y = np.load(f"data/CovariateRandomShift/{st}_Y.npy")
    para = np.load(f"data/CovariateRandomShift/{st}_para.npy")
    return X, Y, para

if __name__ == '__main__':
    pass
    # X, Y, para = Xshift_1(500, [-10,-5,0,5,10], fun_1)
    # SaveData_1(X, Y, para, "FULL")
    # SaveData_1(X[:, 2:], Y[:, 2:], para[:, 2:], "PART")
    # X, Y, para = Xshift_1(50, [-10, -10, -5, -5, 0, 5, 5, 10, 10], fun_1)
    # SaveData_1(X, Y, para, "SPARSEFULL")
    # SaveData_1(X[:, 4:], Y[:, 4:], para[:, 4:], "SPARSEPART")
    # X, Y, para = Xshift_1(300, [0] + [12] * 29, fun_1)
    # SaveData_1(X, Y, para, "FARFULL")
    # SaveData_1(X[:, [0]], Y[:, [0]], para[:, [0]], "FARPART")
    # X, Y, para = Xshift_1(40, [0] + [12] * 39, fun_1)
    # SaveData_1(X, Y, para, "FARSPARSEFULL")
    # SaveData_1(X[:, [0]], Y[:, [0]], para[:, [0]], "FARSPARSEPART")
    #######################################################################
    X, Y, para = Xshift_2(500, [-10, -5, 0, 5, 10], fun_2)
    SaveData_2(X, Y, para, "FULL")
    SaveData_2(X[:, 2:], Y[:, 2:], para[:, 2:], "PART")
    X, Y, para = Xshift_2(50, [-10, -10, -5, -5, 0, 5, 5, 10, 10], fun_2)
    SaveData_2(X, Y, para, "SPARSEFULL")
    SaveData_2(X[:, 4:], Y[:, 4:], para[:, 4:], "SPARSEPART")
    X, Y, para = Xshift_2(300, [0] + [12] * 3, fun_2)
    SaveData_2(X, Y, para, "FARFULL")
    SaveData_2(X[:, [0]], Y[:, [0]], para[:, [0]], "FARPART")
    X, Y, para = Xshift_2(40, [0] + [12] * 10, fun_2)
    SaveData_2(X, Y, para, "FARSPARSEFULL")
    SaveData_2(X[:, [0]], Y[:, [0]], para[:, [0]], "FARSPARSEPART")