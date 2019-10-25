from numpy import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

filename = "data2.txt"


def loadDataSet():
    # load the dataset
    data = loadtxt("data2.txt", delimiter=",")
    # 拿到X和y
    y = np.c_[data[:, 2]]
    x = data[:, 0:2]
    return data, x, y


def map_feature(x1, x2):
    '''''
    Maps the two input features to polynomial features.
    Returns a new feature array with more features of
    X1, X2, X1 ** 2, X2 ** 2, X1*X2, X1*X2 ** 2, etc...
    '''
    x1.shape = (x1.size, 1)
    x2.shape = (x2.size, 1)
    degree = 6
    mapped_fea = ones(shape=(x1[:, 0].size, 1))
    for i in range(1, degree + 1):
        for j in range(i + 1):
            r = (x1 ** (i - j))*(x2 ** j)
            mapped_fea = append(mapped_fea, r, axis=1)
    return mapped_fea


# 计算Sigmoid函数
def sigmoid(x):
    '''
    Compute sigmoid function
    '''
    den = 1.0 + exp(-1.0*x)
    gz = 1.0/den
    return gz


# 定义损失函数
def costfunctionreg(theta, x, y, l):
    m = y.size
    h = sigmoid(x.dot(theta))
    j = -1.0 * (1.0 / m) * (np.log(h).T.dot(y) + np.log(1 - h).T.dot(1 - y)) + (l / (2.0 * m)) * np.sum(np.square(theta[1:]))
    if np.isnan(j[0]):
        return np.inf
    return j[0]


# 计算梯度
def compute_grad(theta, x, y, l):
    m = y.size
    h = sigmoid(x.dot(theta.reshape(-1, 1)))
    grad = (1.0 / m) * x.T.dot(h - y) + (l / m) * np.r_[[[0]], theta[1:].reshape(-1, 1)]
    return grad.flatten()


# 梯度下降并优化
def gradient_descent(xx, y, l):
    initial_theta = np.zeros(xx.shape[1])
    cost = costfunctionreg(initial_theta, xx, y, l)
    print('Cost: \n', cost)
    # 最优化 costfunctionreg
    res2 = minimize(costfunctionreg, initial_theta, args=(xx, y, l), jac=compute_grad, options={'maxiter': 3000})
    return res2


def plotbestfit(data, res2, x, accuracy, l, axes):  # 画出最终分类的图
    # 对X,y的散列绘图
    plotdata(data, 'Microchip Test 1', 'Microchip Test 2', 'y = 1', 'y = 0', axes=None)
    # 画出决策边界
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max(),
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max(),
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    h = sigmoid(map_feature(xx1.ravel(), xx2.ravel()).dot(res2.x))
    h = h.reshape(xx1.shape)
    if axes == None:
        axes = plt.gca()
    axes.contour(xx1, xx2, h, [0.5], linewidths=1, colors='g')
    axes.set_title('Train accuracy {}% with Lambda = {}'.format(np.round(accuracy, decimals=2), l))
    plt.show()


def plotdata(data, label_x, label_y, label_pos, label_neg, axes):
    # 获得正负样本的下标(即哪些是正样本，哪些是负样本)
    neg = data[:, 2] == 0
    pos = data[:, 2] == 1
    if axes == None:
        axes = plt.gca()
    axes.scatter(data[pos][:, 0], data[pos][:, 1], marker='+', c='k', s=60, linewidth=2, label=label_pos)
    axes.scatter(data[neg][:, 0], data[neg][:, 1], c='y', s=60, label=label_neg)
    axes.set_xlabel(label_x)
    axes.set_ylabel(label_y)
    axes.legend(frameon=True, fancybox=True)


def predict(theta, x):
    '''''Predict whether the label
    is 0 or 1 using learned logistic
    regression parameters '''
    m, n = x.shape
    p = zeros(shape=(m, 1))
    h = sigmoid(x.dot(theta.T))
    for it in range(0, h.shape[0]):
        if h[it] > 0.5:
            p[it, 0] = 1
        else:
            p[it, 0] = 0
    return p


def main():
    data, x, y = loadDataSet()
    # 对给定的两个feature做一个多项式特征的映射
    mapped_fea = map_feature(x[:, 0], x[:, 1])

    # 决策边界，咱们分别来看看正则化系数lambda太大太小分别会出现什么情况
    # Lambda = 0 : 就是没有正则化，这样的话，就过拟合咯
    # Lambda = 1 : 这才是正确的打开方式
    # Lambda = 100 : 卧槽，正则化项太激进，导致基本就没拟合出决策边界
    l = 1

    res = gradient_descent(mapped_fea, y, l)
    print(res)

    # 准确率
    accuracy = y[where(predict(res.x, mapped_fea) == y)].size / float(y.size)*100.0
    # 画决策边界
    plotbestfit(data, res, x, accuracy, l, axes=None)


if __name__ == '__main__':
    main()
