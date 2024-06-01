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

h=0.5

def calScore(X, Y, predictor):
    """
    :param X: ndarray, n*k
    :param Y: ndarray, n*k
    :param predictor: any function, tensor -> tensor
    :return: ndarray, S n*k
    """
    Yhat = predictor(torch.tensor(X.reshape(-1,1)).float())\
        .reshape((Y.shape[0],-1)).detach().numpy()
    return np.abs(Yhat-Y)

def simpleGaussianKernel(X1, X2, h):
    """
    Gaussian kernel with bandwidth h, X1 X2 should be of the same type
    ndarray*ndarray -> ndarray;
    float*float -> float;
    ndarray*float -> ndarray;(shape like ndarray)
    :param X1: ndarray length n or float
    :param X2: ndarray length n or float
    :param h: bandwidth
    :return: exp(-(x1-x2)^2/h^2)/h/sqrt(2pi)
    """
    return np.exp(-np.square(X1-X2)/h**2/2)/(h*np.sqrt(2*np.pi))

def calWeight(X, S, newX, h):
    """
    Calculate entire P mtrix
    :param X: ndarray n*k
    :param S: ndarray n*k
    :param h: bandwidth
    :return: W n*(nk+1) ndarray,
            newW ndarray length [nk+1],
            XArray 1*(nk+1) ndarray for W's and newW's column index,
            SArray 1*(nk) ndarray
    """
    XArray = np.zeros((1,X.shape[0]*X.shape[1]+1))
    SArray = S.reshape((1,-1))
    XArray[0, :-1] = X.reshape((1,-1))
    XArray[0, -1] = newX
    X1 = X[:,0].reshape((-1,1))
    return simpleGaussianKernel(X1, XArray, h),\
           simpleGaussianKernel(newX, XArray, h).reshape(-1),\
           XArray, SArray

def calTheta(X, S, newX, h):
    """
    Calculate theta_1^1,cdots,theta_n^1, based on the first column of X
    Theta be \theta_i^1 for i,
    tildeTheta be \tilde{\theta}_i^k for i,k,
    Padd0 be p_{n_1+1,i}^{1,k} for i,k,
    Padd1 be p_{n_1+1,n_1+1}^{1,1}
    :param X: ndarray n*k
    :param h: bandwidth
    :return: Theta ndarray length [n+1], tildeTheta ndarray n*k, Padd0 ndarray n*k, Padd1 float
    """
    W, newW, XArray, SArray = calWeight(X, S, newX, h)
    W_ = W.copy()   # n*(nk+1)
    W_ = (W_ / W_.sum(1).reshape((-1,1)))[:,:-1]    # n*(nk)
    W_ = (W_ / W_.sum(1).reshape((-1,1)))   ###^^^
    W_[SArray>=S[:,[0]]] = 0
    Theta = np.zeros(X.shape[0]+1)
    Theta[:-1] = W_.sum(1)
    newW = newW / newW.sum()
    # \theta_{n_1+1}^1=\tilde{\theta}_{n_1+1}^1
    Theta[-1] = newW[:-1].sum()
    newW_ = newW[:-1].copy()#^
    newW_ = newW_ / newW_.sum()###^^^
    Sord = SArray.argsort().reshape(-1)
    newW_ = newW_[Sord]
    newW_ = np.cumsum(newW_) - newW_
    tildeTheta = newW_[Sord.argsort()].reshape((S.shape[0],-1))
    return Theta, tildeTheta, newW[:-1].reshape((S.shape[0],-1)), newW[-1]

def setSplit(Theta, tildeTheta, Padd0, Padd1, S):
    tildetheta = np.zeros_like(Theta)
    tildetheta[:-1] = tildeTheta[:, 0]
    tildetheta[-1] = Theta[-1]
    padd = np.zeros_like(Theta)
    padd[:-1] = Padd0[:, 0]
    padd[-1] = Padd1
    A1 = tildetheta > (Theta+padd)
    A2 = Theta >= tildetheta
    A3 = ~(A1|A2)
    A2[-1] = False
    A3[-1] = False
    A1v = np.sort((Theta+padd)[A1])
    A2v1 = np.sort(S[:,0][A2[:-1]])
    A2v2 = np.sort(Theta[A2])
    A3v = np.sort(S[:,0][A3[:-1]])
    return A1, A2, A3, A1v, A2v1, A2v2, A3v

def FindS(S, tildeTheta, A1v, A2v1, A2v2, A3v, n, alpha):
    s = S.reshape(-1)
    t = tildeTheta.reshape(-1)
    smask = s.argsort()
    s = s[smask]
    t = t[smask]
    i = 1
    lastS = s[i]
    lastT = t[i]
    while True:
        m1 = (lastT > A1v).sum()
        m2 = ((lastS > A2v1)&(lastT > A2v2)).sum()
        m3 = (lastS > A3v).sum()
        if (m1+m2+m3)/(n+1)>alpha:
            lastS = s[i-1]
            lastT = t[i-1]
            break
        else:
            i += 1
            if i >= len(s):
                return s[-1]
            lastS = s[i]
            lastT = t[i]
    return lastS

def predictConf(predict, s):
    return np.vstack([predict-s,predict+s])

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
    plt.show()

if __name__ == '__main__':
    st = 'FULL'
    X, Y, para = LoadData1(st)
    predictor = Predictor([16, 8])
    predictor.train(X.reshape(-1, 1), Y.reshape(-1, 1), epochs=300)
    X0 = np.linspace(-5, 5, 50)
    st = 'FARFULL'
    X, Y, para = LoadData1(st)
    """
    X=X.reshape((-1,1))
    Y=Y.reshape((-1,1))
    para=para[:,[0]]
    """
    S = calScore(X, Y, predictor)
    conf0 = np.zeros((2,50))
    conf1 = np.zeros((2,50))
    num = X.shape[0]
    for i in range(len(X0)):
        newX = X0[i]
        Theta, tildeTheta, Padd0, Padd1 = calTheta(X, S, newX, h)
        A1, A2, A3, A1v, A2v1, A2v2, A3v = \
            setSplit(Theta, tildeTheta, Padd0, Padd1, S)
        s = FindS(S, tildeTheta, A1v, A2v1, A2v2, A3v, num, 0.9)
        conf0[:, i] = trueConf1(newX).reshape(-1)
        conf1[:, i] = predictConf(predictor(torch. \
            tensor([[newX]]).float()).detach().item(),s).reshape(-1)
    confLen1 = conf0[1,:]-conf0[0,:]
    confLen2 = conf1[1,:]-conf1[0,:]
    Draw(X, Y, X0, conf0, conf1, para, st)