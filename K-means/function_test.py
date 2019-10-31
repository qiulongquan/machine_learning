import numpy as np


def asmatrix():
    # 数组导入，然后返回一个矩阵，值不变
    x = np.array([[1, 2], [3, 4]])
    print(x)
    print(type(x))
    # <class 'numpy.ndarray'>
    m = np.asmatrix(x)
    x[0, 0] = 5
    print(m)
    print(type(m))
    # <class 'numpy.matrix'>

    # matrix([[5, 2],
    #         [3, 4]])


def main():
    asmatrix()


if __name__ == '__main__':
    main()
