import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
# 等高线画法
# https://www.pynote.info/entry/matplotlib-contour

# def f(x, y):
#     return x ** 2 + y ** 2
#
# # X, Y = np.mgrid[-10:11, -10:11]
# X, Y = np.mgrid[-2:2, -2:2]
# print(X)
# # 配列転置
# # print(Y.T)
# print(Y)
#
# Z = f(X, Y)
# print(Z)
#
# # 描画範囲を10段階に分けるように等高線を作成する。
# fig, ax = plt.subplots(figsize=(6, 6), facecolor="w")
# ax.contour(X, Y, Z, levels=5)
# Figureタイトルを設定
# fig.suptitle("contour Functions", fontsize=20)
# Axesのタイトルの設定
# ax.set_title("Contours", size=20, color="red")
# ax.set_xlabel("x", size=20)
# ax.set_ylabel("y", size=20)
# plt.show()

# -------------------------------------------------------------------


def f(x, y):
    return x ** 2 + y ** 2


X, Y = np.mgrid[-12:12, -12:12]
Z = f(X, Y)

# f(x, y) = 0, 10, 50, 100, 200 となる等高線を作成する。
fig, ax = plt.subplots(figsize=(7, 7), facecolor="w")
ax.contour(X, Y, Z, levels=[0, 10, 50, 100, 200])
# Figureタイトルを設定
# fig.suptitle("contour Functions", fontsize=20)
# Axesのタイトルの設定
ax.set_title("Contours", size=20, color="red")
ax.set_xlabel("x", size=20)
ax.set_ylabel("y", size=20)
plt.show()
