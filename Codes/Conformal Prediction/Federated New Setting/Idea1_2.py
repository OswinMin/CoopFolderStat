from Idea1_2_Gen_Pred import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import scipy.stats as stats
import matplotlib.pyplot as plt
import hashlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors

h = 0.5


def sha256(input_string):
    sha256 = hashlib.sha256()
    sha256.update(input_string.encode('utf-8'))
    hash_value = sha256.hexdigest()
    return hash_value

def calScore(X, Y, predictor):
    """
    Calculate score using predictor |Y-f(X)|
    :param X: n*k
    :param Y: n*k
    :param predictor: X -> Yhat
    :return: n*k
    """
    Yhat = predictor(torch.tensor(X).float())
    return torch.abs(Yhat-torch.tensor(Y)).detach().numpy()

def GaussianKernel(X1, X2, h):
    """
    Gaussian kernel with bandwidth h
    :param X1: ndarray length n
    :param X2: ndarray length n
    :param h: bandwidth
    :return: exp(-(x1-x2)^2/h^2)/h/sqrt(2pi)
    """
    return np.exp(-np.square(X1-X2)/h**2/2)/(h*np.sqrt(2*np.pi))

def ReSampling(X, h):
    """
    Given X and gaussian kernel H(,) sample a tilde X
    :param X: X new test point
    :param h: bandwidth for kernel function
    :return: a new sythesized point
    """
    class cd(stats.rv_continuous):
        def _pdf(self, x, *args):
            return GaussianKernel(x, X, h)
    cd_ = cd(a=X-ep*np.abs(X)*5, b=X+ep*np.abs(X)*5)
    samples = cd_.rvs(size=1)
    return samples

def Weight(Xtilde, X, h):
    """
    To generate weight based on Xtilde
    :param Xtilde: new sythesized data point
    :param X: data matrix ndarray n*k
    :param h: bandwidth of kernel function
    :return: weight on each point of X, n*k
    """
    weight = GaussianKernel(np.ones(X.shape[0]*X.shape[1])*Xtilde, X.reshape(-1), h)
    weight = weight / np.sum(weight)
    return weight.reshape(X.shape[0], -1)

def ECDFQ(Z:np.ndarray, P:np.ndarray, alpha):
    """
    Find upper alpha quantile of ECDF at Z with probability P
    :param Z: Value point ndarray [n]
    :param P: Probability respectively ndarray [n]
    :param alpha: upper quantile
    :return:  quantile location, float
    """
    sorted_indices = np.argsort(Z)
    sortedZ = Z[sorted_indices]
    sortedP = P[sorted_indices]
    cumulativeP = np.cumsum(sortedP)
    return sortedZ[np.searchsorted(cumulativeP, 1-alpha)]

def Conf(Yhat, Qtilde):
    return Yhat-Qtilde, Yhat+Qtilde

def Test(X0:np.ndarray, X, Y, theta, predictor):
    """
    Test conformal set on X
    :param X0: test point ndarray [m], from distribution of X first column
    :param X: n*k
    :param Y: n*k
    :param theta: k
    :param predictor: X -> Yhat
    :return:
    """
    Xtilde = [ReSampling(X0[i], h)[0] for i in range(len(X0))]
    lowerbound0, upperbound0 = trueConf(X0, theta[0])
    lowerbound1, upperbound1 = np.zeros_like(lowerbound0), np.zeros_like(lowerbound0)
    P = np.zeros(len(Xtilde)) # conditional coverage
    Y0hat = predictor(torch.tensor(X0).reshape(-1, 1).float()).detach().numpy().reshape(-1)
    Score = calScore(X.reshape(-1, 1), Y.reshape(-1, 1), predictor).reshape(X.shape[0], -1)
    for i in range(len(X0)):
        weight = Weight(Xtilde[i], X, h)
        Qtilde = ECDFQ(Score.reshape(-1), weight.reshape(-1), 0.1)
        lowerbound1[i], upperbound1[i] = Conf(Y0hat[i], Qtilde)
        P[i] = stats.norm.cdf(lowerbound1[i], loc=X0[i]**2, scale=ep*(np.abs(X0[i])+theta[0]))\
            - stats.norm.cdf(upperbound1[i], loc=X0[i]**2, scale=ep*(np.abs(X0[i])+theta[0]))
    return lowerbound0, upperbound0, lowerbound1, upperbound1, P

def Draw(X, Y, X0, loc, theta, lowerbound0, upperbound0, lowerbound1, upperbound1):
    cmap = cm.get_cmap('cividis')
    st = f"loc: " + ', '.join([str(i) for i in loc]) + f';  num: {num}' + '\ntheta: ' + \
         ', '.join([str(round(i, 1)) for i in theta])
    plt.figure(figsize=(8,6))
    if len(X.shape)==1:
        X_ = X.reshape((-1,1))
        Y_ = Y.reshape((-1,1))
    else:
        X_ = X
        Y_ = Y
    norm = mcolors.Normalize(vmin=-X_.shape[1]-1, vmax=X_.shape[1]-1)
    for j in range(X_.shape[1]):
        x_ = X_[:,j][(X_[:,j]>min(X0))&(X_[:,j]<max(X0))]
        y_ = Y_[:,j][(X_[:,j]>min(X0))&(X_[:,j]<max(X0))]
        plt.scatter(x_, y_, s=3, label=f"loc: {loc[j]}", color=cmap(norm(j)))
    plt.plot(X0, lowerbound0, color='red', label='True')
    plt.plot(X0, upperbound0, color='red')
    plt.plot(X0, lowerbound1, color='blue', label='Estimated')
    plt.plot(X0, upperbound1, color='blue')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(st)
    plt.savefig(f"resultFig/Idea1_2/{sha256(st)[:5]}.png", dpi=400)
    plt.show()

if __name__ == '__main__':
    h = 0.5
    loc = [0, 20]
    num = 500
    X, Y, theta = Xshift(num, loc, fun)
    predictor = Predictor([18, 9])
    predictor.train(X.reshape(-1, 1), Y.reshape(-1, 1), epochs=300)
    X0 = np.linspace(-4, 4, 500)
    lowerbound0, upperbound0, lowerbound1, upperbound1, P = Test(X0, X, Y, theta, predictor)
    Draw(X, Y, X0, loc, theta, lowerbound0, upperbound0, lowerbound1, upperbound1)

    ### 用上面生成的 X 第一列
    X_, Y_ = X[:, [0]], Y[:, [0]]
    X0 = np.linspace(-4, 4, 500)
    predictor = Predictor([15, 7])
    predictor.train(X_.reshape(-1, 1), Y_.reshape(-1, 1), epochs=200)
    lowerbound0, upperbound0, lowerbound1, upperbound1, P = Test(X0, X_, Y_, theta[[0]], predictor)
    Draw(X_, Y_, X0, [loc[0]], theta[[0]], lowerbound0, upperbound0, lowerbound1, upperbound1)