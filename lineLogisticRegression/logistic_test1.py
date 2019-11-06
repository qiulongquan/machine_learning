from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

'''
下载load_iris数据然后显示出来

通过下载下来的数据进行梯度下降的训练，得到最小损失函数值，并显示出来

原文链接：https://blog.csdn.net/huakai16/article/details/78140304
'''


class logistic(object):
    def __init__(self):
        self.W = None

    def train(self, x, y, learn_rate=0.01, num_iters=5000):
        num_train, num_feature = x.shape
        # init the weight
        self.W = 0.001 * np.random.randn(num_feature, 1).reshape((-1, 1))
        loss = []

        for i in range(num_iters):
            error, dw = self.compute_loss(x, y)
            self.W += -learn_rate * dw

            loss.append(error)
            if i % 200 == 0:
                print("i={},error={}".format(i, error))
        return loss

    def compute_loss(self, x, y):
        num_train = x.shape[0]
        h = self.output(x)
        loss = -np.sum((y * np.log(h) + (1 - y) * np.log((1 - h))))
        loss = loss / num_train

        dw = x.T.dot((h - y)) / num_train

        return loss, dw

    def output(self, x):
        g = np.dot(x, self.W)
        return self.sigmod(g)

    def sigmod(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, x_test):
        h = self.output(x_test)
        y_pred = np.where(h >= 0.5, 1, 0)
        return y_pred


def show_test():
    # 通过sklearn包调用装载iris数据
    iris = load_iris()
    # 提取isir里面data数据集
    data = iris.data
    print(data[0:100, [0, 2]])
    target = iris.target
    # print data[:10]
    # print target[10:]
    x = data[0:100, [0, 2]]
    y = target[0:100]
    print
    x[:5]
    print
    y[-5:]
    label = np.array(y)
    index_0 = np.where(label == 0)
    # 画出x和o的各个分布的点
    plt.scatter(x[index_0, 0], x[index_0, 1], marker='x', color='b', label='0', s=15)
    index_1 = np.where(label == 1)
    plt.scatter(x[index_1, 0], x[index_1, 1], marker='o', color='r', label='1', s=15)

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend(loc='upper left')
    plt.show()


def show_test1():
    # 通过sklearn包调用装载iris数据
    iris = load_iris()
    # 提取isir里面data数据集
    data = iris.data
    print(data[0:100, [0, 2]])
    target = iris.target
    # print data[:10]
    # print target[10:]
    x = data[0:100, [0, 2]]
    y = target[0:100]
    y = y.reshape((-1, 1))
    # add the x0=1
    one = np.ones((x.shape[0], 1))
    print("one type={}".format(type(one)))
    x_train = np.hstack((one, x))
    classify = logistic()
    loss = classify.train(x_train, y)
    print("classify.W=", classify.W)
    plt.plot(loss)
    plt.xlabel('Iteration number')
    plt.ylabel('Loss value')
    plt.show()


def show_test2():
    # 通过sklearn包调用装载iris数据
    iris = load_iris()
    # 提取isir里面data数据集
    data = iris.data
    print(data[0:100, [0, 2]])
    target = iris.target
    # print data[:10]
    # print target[10:]
    x = data[0:100, [0, 2]]
    y = target[0:100]
    y = y.reshape((-1, 1))
    # add the x0=1
    one = np.ones((x.shape[0], 1))
    print("one type={}".format(type(one)))
    x_train = np.hstack((one, x))
    classify = logistic()
    loss = classify.train(x_train, y)
    print("classify.W=", classify.W)
    # plt.plot(loss)
    # plt.xlabel('Iteration number')
    # plt.ylabel('Loss value')
    # plt.show()

    # 下面的是画y=1和y=0的所有的点
    label = np.array(y)
    index_0 = np.where(label == 0)
    plt.scatter(x[index_0, 0], x[index_0, 1], marker='x', color='b', label='0', s=15)
    index_1 = np.where(label == 1)
    plt.scatter(x[index_1, 0], x[index_1, 1], marker='o', color='r', label='1', s=15)

    # 显示决策边界
    # show the decision boundary
    # 通过下面的2个句子来执行决策边界判定
    # 这个地方还不是很明白 ？
    print(classify.W[0])
    print(classify.W[1])
    print(classify.W[2])

    x1 = np.arange(4, 7.5, 0.5)
    x2 = (- classify.W[0] - classify.W[1] * x1) / classify.W[2]

    plt.plot(x1, x2, color='black')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend(loc='upper left')
    plt.show()


def main():
    # show_test()
    # show_test1()
    show_test2()


if __name__ == '__main__':
    main()
