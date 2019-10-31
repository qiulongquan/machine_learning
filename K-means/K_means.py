from numpy import *
import xlrd
import matplotlib.pyplot as plt


# 计算欧氏距离
def euclDistance(vector1, vector2):
    '''
    :param vector1: 第j个均值向量
    :param vector2: 第i个样本
    :return: 距离值
    '''
    return sqrt(sum(power(vector2 - vector1, 2)))


# init centroids with random samples
def initCentroids(dataSet, k):
    '''
    :param dataSet: 数据集
    :param k: 需要聚类的个数
    :return:  返回k个均值向量
    '''
    numSamples, dim = dataSet.shape
    centroids = zeros((k, dim))
    for i in range(k):
        index = int(random.uniform(0, numSamples))
        centroids[i, :] = dataSet[index, :]
    return centroids


# k-means cluster
def kmeans(dataSet, k):
    '''
    :param dataSet: 数据集
    :param k:  需要聚类的个数
    :return:
    '''
    # 样本的个数
    numSamples = dataSet.shape[0]
    # 第一列存储该样本所属的集群
    # 第二列存储此样本与其质心之间的误差
    clusterAssment = mat(zeros((numSamples, 2)))
    clusterChanged = True

    ## step 1:从数据集中随机选择k个样本作为初始均值向量
    centroids = initCentroids(dataSet, k)

    while clusterChanged:
        clusterChanged = False
        ## 循环每一个样本
        for i in range(numSamples):
            minDist = 100000.0  #存放最短的距离
            minIndex = 0     # 第i个样本的簇标记
            ## 循环每一个均值向量
            ## step 2: 找到第i个样本的最近的均值向量
            for j in range(k):
                distance = euclDistance(centroids[j, :], dataSet[i, :])
                if distance < minDist:
                    minDist = distance
                    minIndex = j

                    ## step 3: 更新第i个样本的簇标记和误差
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist ** 2

                ## step 4: 更新均值向量
        for j in range(k):
            pointsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == j)[0]]
            centroids[j, :] = mean(pointsInCluster, axis=0)

    print('Congratulations, cluster complete!')
    return centroids, clusterAssment


# show your cluster only available with 2-D data
def showCluster(dataSet, k, centroids, clusterAssment):
    numSamples, dim = dataSet.shape
    if dim != 2:
        print ("Sorry! I can not draw because the dimension of your data is not 2!")
        return 1

    # 同类簇的颜色和形状，必须和下面的中心点的顺序一致
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    print("K={}".format(k))
    if k > len(mark):
        print("Sorry! Your k is too large! please contact Zouxy")
        return 1

        # draw all samples
    for i in range(numSamples):
        markIndex = int(clusterAssment[i, 0])
        # 画同类簇的图形
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

    # 中心点的颜色和形状，必须和上面的同类簇的顺序一致
    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # draw the centroids
    for i in range(k):
        # 画中心点图形
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=15)
    plt.show()


def main():
    ## step 1: load data
    print("step 1: load data...")
    dataSet = []
    # 读取excel文档的类库
    data = xlrd.open_workbook('watermelon4.0.xlsx')
    # 确定读取第一页数据
    table = data.sheets()[0]
    for line in range(0, table.nrows):
        lineArr = table.row_values(line)
        dataSet.append([float(lineArr[0]), float(lineArr[1])])

    ## step 2: clustering...
    print("step 2: clustering...")
    # 数组导入，然后返回一个矩阵，值不变
    dataSet = mat(dataSet)
    k = 3
    centroids, clusterAssment = kmeans(dataSet, k)

    ## step 3: show the result
    print("step 3: show the result...")
    showCluster(dataSet, k, centroids, clusterAssment)


if __name__ == '__main__':
    main()
