from numpy import loadtxt,where
from pylab import scatter, show, legend, xlabel, ylabel

#load the dataset
data = loadtxt("data1.txt", delimiter=",")
#可以看出数据是一个二维数组，维度是100*3
print(data)

X = data[:,0:2]
#X存放的是数据的特征，维度是：100*2
# print(X.shape)
y = data[:, 2]
#y存放的是数据的标签，维度是：100*1
# print(y)

pos = where(y == 1)
#pos是y中数据等于1的下标索引
# print(pos)

neg = where(y==0)
#neg是y中数据等于0的下标索引
# print(neg)

# pos代表y=1的坐标集合，X[pos,0]获取的是每一个y=1的坐标点的横坐标
# X[pos, 1]获取的是每一个y=1的坐标点的纵坐标
print("pos X:",X[pos,0])
print("pos Y:",X[pos,1])

#python中数据可视化函数scatter(数据的横坐标向量，数据的纵坐标向量，marker='0'数据以点的形式显示，c='b'数据点是blue颜色)
scatter(X[pos,0],X[pos, 1],marker='o', c='b')
scatter(X[neg,0],X[neg, 1],marker='x', c='r')

#二维坐标中的横坐标文本
xlabel("Feature1/Exam 1 score")
#二维坐标中的纵坐标文本
ylabel("Feature2/Exam 2 score")
#说明二维坐标中o表示Pass,x表示Fail
legend(["Pass","Fail"])
show()