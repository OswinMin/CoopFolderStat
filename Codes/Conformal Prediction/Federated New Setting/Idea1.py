from Idea1_Gen_Pred import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import scipy.stats as stats
import matplotlib.pyplot as plt

h = 0.5

def calScore(X, Y, predictor):
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
    cd_ = cd()
    samples = cd_.rvs(size=1)
    return samples

def Weight(Xtilde, X, h):
    """
    To generate weight based on Xtilde
    :param Xtilde: new sythesized data point
    :param X: data matrix ndarray n*k
    :param h: bandwidth of kernel function
    :return: weight on each point of X
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
    :return:  quantile location
    """
    sorted_indices = np.argsort(Z)
    sortedZ = Z[sorted_indices]
    sortedP = P[sorted_indices]
    cumulativeP = np.cumsum(sortedP)
    return sortedZ[np.searchsorted(cumulativeP, 1-alpha)]

def Conf(Yhat, Qtilde):
    return [Yhat-Qtilde,Yhat+Qtilde]

def Test(X0:np.ndarray, X, Y, predictor):
    """
    Test conformal set on X
    :param X0: test point ndarray [n]
    :return: Conf0(true) Conf1(predicted) P(true coverage)
    """
    Xtilde = [ReSampling(X0[i], h)[0] for i in range(len(X0))]
    Conf0 = trueConf(X0)
    Conf1 = np.zeros_like(Conf0)
    P = np.zeros(len(Xtilde))
    Y0hat = predictor(torch.tensor(X0).reshape(-1, 1).float()).detach().numpy().reshape(-1)
    Score = calScore(X.reshape(-1, 1), Y.reshape(-1, 1), predictor).reshape(X.shape[0], -1)
    for i in range(len(X0)):
        weight = Weight(Xtilde[i], X, h)
        Qtilde = ECDFQ(Score.reshape(-1), weight.reshape(-1), 0.1)
        Conf1[:,i] = Conf(Y0hat[i], Qtilde)
        P[i] = stats.norm.cdf(Conf1[1,i], loc=X0[i]**2, scale=ep*np.abs(X0[i]))\
            - stats.norm.cdf(Conf1[0,i], loc=X0[i]**2, scale=ep*np.abs(X0[i]))
    return Conf0, Conf1, P

def Draw(X, Y, X0, Conf0, Conf1):
    plt.figure(figsize=(8,6))
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    X_ = X[(X>min(X0))&(X<max(X0))]
    Y_ = Y[(X>min(X0))&(X<max(X0))]
    plt.scatter(X_, Y_, s=0.5, label="Training Data")
    plt.plot(X0, Conf0[0,:], color='red')
    plt.plot(X0, Conf0[1,:], color='red')
    plt.plot(X0, Conf1[0,:], color='blue')
    plt.plot(X0, Conf1[1,:], color='blue')
    plt.show()

if __name__ == '__main__':
    h = 0.5
    X, Y = Xshift(500, [0, 0.1, 0.2, -0.05, 0.6, -1], fun)
    predictor = Predictor()
    predictor.train(X.reshape(-1, 1), Y.reshape(-1, 1), epochs=200)
    X0 = np.linspace(-3,3,500)
    Conf0 = []
    Conf1 = []
    P = []
    for i in range(10):
        Conf0_, Conf1_, P_ = Test(X0, X, Y, predictor)
        Conf0.append(Conf0_)
        Conf1.append(Conf1_)
        P.append(P_)
    Conf0_ = np.zeros_like(Conf0_)
    Conf1_ = np.zeros_like(Conf1_)
    for i in range(len(Conf0)):
        Conf0_ += Conf0[i]
        Conf1_ += Conf1[i]
    Conf0_ /= len(Conf0)
    Conf1_ /= len(Conf1)
    Draw(X, Y, X0, Conf0_, Conf1_)