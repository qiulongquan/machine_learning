"""
test例子，需要查看那个例子就去掉注释

load_data_set
判断这两种数据类型是否类型一样，值一样，维度一样
y = data_set[:5, 0:2]
y1 = np.c_[data_set[:5, 0:2]]
"""

from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

filename = "data2.txt"


def load_data_set_test():
    data_set = loadtxt(filename, delimiter=",")
    y = data_set[:5, 0:2]
    y1 = np.c_[data_set[:5, 0:2]]
    print(y)
    print("="*50)
    print(y1)
    print(y.shape)
    print(y1.shape)
    print(type(y))
    print(type(y1))
    print(type(y1) == type(y))
    print("y scalar:", np.isnan(y[0]))
    print("y1 scalar:", np.isnan(y1[0]))


def numpy_c_function_test():
    # numpy中np.c_和np.r_
    # np.r_是按列连接两个矩阵，就是把两矩阵行合并，要求行数相等，类似于pandas中的concat()。
    # np.c_是按行连接两个矩阵，就是把两矩阵列合并，要求列数相等，类似于pandas中的merge()。
    # https: // blog.csdn.net / yj1556492839 / article / details / 79031693
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    c = np.c_[a, b]

    print(np.r_[a, b])
    print(c)
    print(np.c_[c, a])


def set_shape_text():
    # 获得一个一维数组，然后重新设定shape的维度数
    data_set = loadtxt(filename, delimiter=",")
    y = data_set[:12, 0]
    print(y)
    print(y.shape)
    y.shape = (3, 4)
    print(y)
    print(y.shape)

    print("取的是y数组的每一列的长度。y[:, 1].size:", y[:, 1].size)
    # 设定ones数组的维度数，ones的数组内的全部值都是1
    mapped_fea = ones(shape=(y[:, 0].size, 1))
    print(mapped_fea)
    print(mapped_fea.shape)


def range_test():
    x1 = np.array([1, 2, 3, 4, 5])
    x2 = np.array([2, 4, 6, 8, 10])
    print(type(x1))
    print(type(x2))
    x1.shape = (x1.size, 1)
    x2.shape = (x2.size, 1)
    mapped_fea = ones(shape=(x1[:, 0].size, 1))
    # 初始状态5行1列全部都是1
    print(mapped_fea)
    degree = 6
    # i范围在1到6 不包括7
    for i in range(1, degree + 1):
        # j范围在0到6 不包括7
        for j in range(i + 1):
            r = (x1 ** (i - j)) * (x2 ** j)
            print("i={},j={},r={}".format(i, j, r))
            mapped_fea = append(mapped_fea, r, axis=1)
            print(mapped_fea)
        print("="*50)


def sigmoid(x_start, x_end):
    """
    Compute sigmoid function
    """
    dot_count = 100
    x_array = linspace(x_start, x_end, dot_count)
    y_array = []
    for x in x_array:
        den = 1.0 + exp(-1.0*x)
        gz = 1.0/den
        y_array.append(gz)
    plt.plot(x_array, y_array, color="red", linewidth=2)
    plt.show()


def zeros_test():
    # 二维全部都是0的数组
    x = np.zeros((4, 2))
    # 三维全部都是0的数组
    # x = np.zeros((4, 3, 2))
    print(x)
    print(x.shape)


def dot_multiplication():
    array_a = np.array([1, 2, 3, 4])
    array_a.shape = (2, 2)
    array_b = np.array([2, 3, 4, 5])
    array_b.shape = (2, 2)
    # dot矩阵相乘
    array_c = array_a.dot(array_b)
    print(array_c)
    print(array_c.shape)


def main():
    # test例子，需要查看那个例子就去掉注释
    # load_data_set_test()
    # numpy_c_function_test()
    # set_shape_text()
    # range_test()
    # sigmoid(-5, 5)
    # zeros_test()
    dot_multiplication()


if __name__ == '__main__':
    main()
