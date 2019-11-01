from numpy import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

filename = "data2.txt"


def load_data_set():
    # load the dataset
    data_set = loadtxt(filename, delimiter=",")
    # 拿到X和y
    # y = np.c_[data_set[:, 2]]
    y = np.c_[data_set[:, 2]]
    # print(y)
    # print(y.shape)
    x = data_set[:, 0:2]
    # print(x)
    # print(x.shape)
    return data_set, x, y


def map_feature(x1, x2):
    """
    Maps the two input features to polynomial features.
    Returns a new feature array with more features of
    X1, X2, X1 ** 2, X2 ** 2, X1*X2, X1*X2 ** 2, etc...
    """
    # 通过下面重新定义维度数量，将一行X列的数组变成X行1列的数组
    print(type(x1))
    x1.shape = (x1.size, 1)
    x2.shape = (x2.size, 1)
    degree = 6
    # 设定ones数组的维度数，ones的数组内的全部值都是1
    # "取的是y数组的每一列的长度。y[:, 1].size:"
    mapped_fea = ones(shape=(x1[:, 0].size, 1))
    # i范围在1到6 不包括7
    for i in range(1, degree + 1):
        # j范围在0到7 不包括8
        for j in range(i + 1):
            # r函数的结果是一个倍数递归
            r = (x1 ** (i - j))*(x2 ** j)
            mapped_fea = append(mapped_fea, r, axis=1)
    return mapped_fea


# 计算Sigmoid函数
def sigmoid(x):
    """
    Compute sigmoid function
    """
    den = 1.0 + exp(-1.0*x)
    gz = 1.0/den
    return gz


# 定义损失函数
def costfunctionreg(initial_theta, mapped_fea, y, l):
    m = y.size
    # Sigmoid函数，即f(x) = 1 / (1 + e - x)。是神经元的非线性作用函数。
    # h=sigmoid计算完的结果数组
    h = sigmoid(mapped_fea.dot(initial_theta))
    # 神经网络损失函数求导
    # 神经网络的损失函数可以理解为是一个多级的复合函数，求导使用链式法则。
    # https://blog.csdn.net/HHTNAN/article/details/78316785
    # 交叉墒 cross-entropy 动画演示
    # http://neuralnetworksanddeeplearning.com/chap3.html#exercise_35813

    j = -1.0 * (1.0 / m) * (np.log(h).T.dot(y) + np.log(1 - h).T.dot(1 - y)) + (l / (2.0 * m)) * np.sum(np.square(initial_theta[1:]))
    print("j数组", j)
    if np.isnan(j[0]):
        return np.inf
    return j[0]


# 计算梯度
def compute_grad(theta, x, y, l):
    m = y.size
    h = sigmoid(x.dot(theta.reshape(-1, 1)))
    grad = (1.0 / m) * x.T.dot(h - y) + (l / m) * np.r_[[[0]], theta[1:].reshape(-1, 1)]
    return grad.flatten()

    m = y.size
    h = sigmoid(X.dot(theta.reshape(-1, 1)))
    grad = (1.0 / m) * X.T.dot(h - y) + (l / m) * np.r_[[[0]], theta[1:].reshape(-1, 1)]
    return grad.flatten()


# 梯度下降并优化
def gradient_descent(mapped_fea, y, l):
    # 全部都是0的初始数组
    initial_theta = np.zeros(mapped_fea.shape[1])
    cost = costfunctionreg(initial_theta, mapped_fea, y, l)
    print('Cost: \n', cost)
    # 最优化 costfunctionreg 并寻找最小的结果
    res2 = minimize(costfunctionreg, initial_theta, args=(mapped_fea, y, l), jac=compute_grad, options={'maxiter': 3000})
    return res2


def plotbestfit(data_set, res2, x, accuracy, l, axes):  # 画出最终分类的图
    """
        # x 是点的坐标（横坐标 纵坐标）
        # y 是标量 有2种数值 0 和 1
        # data_set 是包括了x和y的所有值全部取得 1.0709,0.10015,0 一共3个值
        # res2 是梯度下降并优化的最终值（最小损失函数值）
        # l 是一个常量参数
        # accuracy 是计算出来的准确率
        # axes默认是None
    """
    # 对X,y的散列绘图
    plotdata(data_set, 'Microchip Test 1', 'Microchip Test 2', 'y = 1', 'y = 0', axes=None)
    # 画出决策边界
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max(),
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max(),
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    h = sigmoid(map_feature(xx1.ravel(), xx2.ravel()).dot(res2.x))
    h = h.reshape(xx1.shape)
    if axes == None:
        axes = plt.gca()
    # 画等高线Contours图像
    axes.contour(xx1, xx2, h, [0.5], linewidths=1, colors='g')
    axes.set_title('Train accuracy {}% with Lambda = {}'.format(np.round(accuracy, decimals=2), l))
    plt.show()


def plotdata(data_set, label_x, label_y, label_pos, label_neg, axes):
    # 获得正负样本的下标(即哪些是正样本，哪些是负样本)
    neg = data_set[:, 2] == 0
    pos = data_set[:, 2] == 1
    if axes == None:
        axes = plt.gca()
    axes.scatter(data_set[pos][:, 0], data_set[pos][:, 1], marker='+', c='k', s=60, linewidth=2, label=label_pos)
    axes.scatter(data_set[neg][:, 0], data_set[neg][:, 1], c='y', s=60, label=label_neg)
    axes.set_xlabel(label_x)
    axes.set_ylabel(label_y)
    axes.legend(frameon=True, fancybox=True)


def predict(theta, mapped_fea):
    """
    Predict whether the label
    is 0 or 1 using learned logistic
    regression parameters
    """
    m, n = mapped_fea.shape
    p = zeros(shape=(m, 1))
    h = sigmoid(mapped_fea.dot(theta.T))
    for it in range(0, h.shape[0]):
        if h[it] > 0.5:
            p[it, 0] = 1
        else:
            p[it, 0] = 0
    return p


def main():
    # x 是点的坐标（横坐标 纵坐标）
    # y是标量 有2种数值 0 和 1
    # data_set 是包括了x和y的所有值全部取得 1.0709,0.10015,0 一共3个值
    data_set, x, y = load_data_set()
    # 对给定的两个feature做一个多项式特征的映射
    mapped_fea = map_feature(x[:, 0], x[:, 1])

    # 决策边界，咱们分别来看看正则化系数lambda太大太小分别会出现什么情况
    # Lambda = 0 : 就是没有正则化，这样的话，就过拟合咯
    # Lambda = 1 : 这才是正确的打开方式
    # Lambda = 100 : 卧槽，正则化项太激进，导致基本就没拟合出决策边界
    l = 1
    # y有2种数值1和0
    res = gradient_descent(mapped_fea, y, l)
    print("res=", res)

    # 准确率
    print("predict(res.x, mapped_fea:{})".format(predict(res.x, mapped_fea)))
    print("where:{}".format(where(predict(res.x, mapped_fea) == y)))
    print("y={}".format(y[where(predict(res.x, mapped_fea) == y)]))
    # >> > a = np.array([2, 4, 6, 8, 10])
    # >> > np.where(a > 5)  # 返回索引
    # (array([2, 3, 4]),)
    # >> > a[np.where(a > 5)]  # 等价于 a[a>5]
    # array([6, 8, 10])
    accuracy = y[where(predict(res.x, mapped_fea) == y)].size / float(y.size)*100.0
    # 画决策边界

    # x 是点的坐标（横坐标 纵坐标）
    # y 是标量 有2种数值 0 和 1
    # data_set 是包括了x和y的所有值全部取得 1.0709,0.10015,0 一共3个值
    # res 是梯度下降并优化的最终值（最小损失函数值）
    # l 是一个常量参数
    # accuracy 是计算出来的准确率
    data_set, x, y = load_data_set()
    plotbestfit(data_set, res, x, accuracy, l, axes=None)


if __name__ == '__main__':
    main()
