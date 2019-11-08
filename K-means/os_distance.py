import numpy as np

"""
欧几里得距离的求解
原文链接：
https://blog.csdn.net/Mr_EvanChen/article/details/77511312
https://blog.csdn.net/Kevin_cc98/article/details/73742037
"""
x = np.random.random(2)
y = np.random.random(2)

x1 = [1, 2]
y1 = [3, 4]
x1 = np.array(x1)
y1 = np.array(y1)

# solution1
dist1 = np.linalg.norm(x - y)

# solution2
dist2 = np.sqrt(np.sum(np.square(x1 - y1)))

print('x1', x1)
print('y1', y1)

print("=" * 50)

# print('dist1', dist1)
print('dist2', dist2)

