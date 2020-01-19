import numpy as np
import matplotlib.pyplot as plt

"""
http://www.isc.meiji.ac.jp/~mizutani/python/print/matplotlib/matplotlib.pdf
関数グラフ 绘画方法
"""
def moving_coordinate():
    x1 = np.random.rand(10)*10
    x1_max = np.max(x1)
    x1_min = np.min(x1)
    x2 = np.random.rand(10)*10
    x2_max = np.max(x2)
    x2_min = np.min(x2)
    # linspace是等分点，不是随机点，不要
    # x1 = np.linspace(-2, 2, 20) * 10
    # x2 = np.linspace(-3, 3, 20) * 10
    print(type(x1))
    plt.scatter(x1, x2, label='before moving')
    print("x1=", x1)
    print("x2=", x2)
    mean1 = 1
    mean2 = 1
    while mean1 > 0 or mean2 > 0:
        sum1 = np.sum(x1)
        mean1 = np.mean(x1, dtype=np.int)
        sum2 = np.sum(x2)
        mean2 = np.mean(x2, dtype=np.int)
        print("mean1={},mean2={}".format(mean1, mean2))
        print("sum1={},sum2={}".format(sum1, sum2))
        x1 = x1 - mean1
        print(type(x1))
        x2 = x2 - mean2
        print("x1={},x2={}".format(x1, x2))
    # xlim,ylim matplotlibでグラフの描画範囲を設定
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    # x 軸 (直線 x=0，采用-来划线，颜色是蓝色，lw代表线的宽度)
    plt.axhline(0, ls='-', c='b', lw=0.5)
    # y 軸 (直線 y=0，采用-来划线，颜色是蓝色，lw代表线的宽度)
    plt.axvline(0, ls='-', c='b', lw=0.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Graphs of Normalizing sample')
    plt.scatter(x1, x2, label='after moving')
    plt.legend()  # 凡例表示
    plt.show()


def show_sin_cos():
    x = np.linspace(-np.pi, np.pi)
    # plt.plot(x, np.cos(x), color='r', ls='-', label='cos')
    plt.plot(x, np.sin(x), color='b', ls='-', label='sin')
    # plt.plot(x, np.tan(x), color='c', marker='s', ls='None', label='tan')

    # xlim,ylim matplotlibでグラフの描画範囲を設定
    # https: // www.pynote.info / entry / matplotlib - xlim - ylim
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-1.5, 1.5)
    # x 軸 (直線 x=0，采用-来划线，颜色是蓝色，lw代表线的宽度)
    plt.axhline(0, ls='-', c='b', lw=0.5)
    # y 軸 (直線 y=0，采用-来划线，颜色是蓝色，lw代表线的宽度)
    plt.axvline(0, ls='-', c='b', lw=0.5)
    plt.legend()  # 凡例表示
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Graphs of Normalizing sample')
    # plt.savefig(’image/func_plot2.png’)
    plt.show()


def main():
    moving_coordinate()
    show_sin_cos()


if __name__=='__main__':
    main()
