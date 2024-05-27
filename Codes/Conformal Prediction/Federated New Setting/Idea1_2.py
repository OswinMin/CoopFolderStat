from Idea1_2_Gen_Pred import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import scipy.stats as stats
import matplotlib.pyplot as plt
import hashlib

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
        Conf1[:,i] = Conf(Y0hat[i], Qtilde)
        P[i] = stats.norm.cdf(Conf1[1,i], loc=X0[i]**2, scale=ep*np.abs(X0[i]))\
            - stats.norm.cdf(Conf1[0,i], loc=X0[i]**2, scale=ep*np.abs(X0[i]))
    return Conf0, Conf1, P