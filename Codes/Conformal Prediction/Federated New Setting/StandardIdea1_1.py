from GenPredGeneral import *
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

"""
True X has covariate shift
True Y is same conditional on X
Any new test X0
draw Xtilde based on H(,X0)
Construct weight based on Xtilde and H(Xtidle,X) placed on Score
"""

def sha256(input_string):
    sha256 = hashlib.sha256()
    sha256.update(input_string.encode('utf-8'))
    hash_value = sha256.hexdigest()
    return hash_value

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
    cd_ = cd(a=X-ep*np.abs(X)*5, b=X+ep*np.abs(X)*5)
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
    Conf0 = trueConf1(X0)
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

def Draw(X, Y, X0, Conf0, Conf1, para, st):
    cmap = cm.get_cmap('brg')
    para_ = np.unique(para, axis=1)
    loc = para[0,:]
    loc_ = para_[0,:]
    sigma = para[1,:]
    sigma_ = para_[1,:]
    num_ = np.zeros(para_.shape[1])
    for i in range(para_.shape[1]):
        num_[i] = X.shape[0]*np.sum((loc==loc_[i])&(sigma==sigma_[i]))
    title = f"    loc: " + ';'.join([f'{int(i):4}' if i.is_integer()
                                    else f'{i:4}' for i in loc_]) + '\n' + \
            f"sigma: " + ';'.join([f'{int(i):4}' if i.is_integer()
                                    else f'{i:4}' for i in sigma_]) + '\n' + \
            f"   num: " + ';'.join([f'{int(i):4}' if i.is_integer()
                                    else f'{i:4}' for i in num_])
    if len(X.shape)==1:
        X = X.reshape((-1,1))
        Y = Y.reshape((-1,1))
    plt.figure(figsize=(8,6))
    norm = mcolors.Normalize(vmin=0, vmax=len(loc_) - 1)
    for j in range(len(loc_)):
        x_ = X[:, (loc==loc_[j])&(sigma==sigma_[j])].reshape(-1)
        y_ = Y[:, (loc==loc_[j])&(sigma==sigma_[j])].reshape(-1)
        y_ = y_[(x_>min(X0))&(x_<max(X0))]
        x_ = x_[(x_>min(X0))&(x_<max(X0))]
        if len(x_) > 0:
            plt.scatter(x_, y_, s=4,
            label=f"({int(loc_[j]) if loc_[j].is_integer() else loc_[j]},"
                f"{int(sigma_[j]) if sigma_[j].is_integer() else sigma_[j]})",
            color=cmap(norm(j)), alpha=0.5)
    plt.plot(X0, Conf0[0,:], color='indianred', label='True', linewidth=2)
    plt.plot(X0, Conf0[1,:], color='indianred', linewidth=2)
    plt.plot(X0, Conf1[0,:], color='slateblue', label='Estimated', linewidth=2)
    plt.plot(X0, Conf1[1,:], color='slateblue', linewidth=2)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.legend()
    plt.savefig(f"resultFig/Idea1_1/{st}.png", dpi=400)
    plt.show()

if __name__ == '__main__':
    h = 1
    st = 'FULL'
    X, Y, para = LoadData1(st)
    predictor = Predictor([16,8])
    predictor.train(X.reshape(-1, 1), Y.reshape(-1, 1), epochs=300)
    X0 = np.linspace(-5,5,100)
    for st in ['FULL','PART','SPARSEFULL','SPARSEPART','FARFULL','FARPART']:
        X, Y, para = LoadData1(st)
        Conf0_, Conf1_, P_ = Test(X0, X, Y, predictor)
        Draw(X, Y, X0, Conf0_, Conf1_, para, st)

