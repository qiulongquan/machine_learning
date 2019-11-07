# machine_learning

机器学习算法代码实现
正在分析 逻辑回归中判定边界是非线性函数 代码

```
K-means/K_means.py
重点是
dataSet
centroids
是怎么算出来的，根据什么条件来区分每一个同类簇

其中y=g(z)函数正是上述sigmoid函数(或者简单叫做S形函数)。这就是逻辑回归中的预测输出函数。
有了预测输出函数，我们考虑一下，为什么逻辑回归就能做到分类问题？其实是逻辑回归根据我们的样本点获得这些数据的判定边界。那什么是判定边界呢？
你可以简单理解为是用以对不同类别的数据分割的边界，边界的两旁应该是不同类别的数据。我们在二维直角坐标系中，来看看具体的判定边界是什么样子：
我们来思考一个问题：逻辑回归是如何根据样本点获得这些判定边界呢？

根据sigmoid函数，我们发现：

当g(z)≥0.5时, z≥0;对于=g(θTX)≥0.5, 则θTX≥0, 此时意味着预估y=1;

反之，当预测y = 0时，θTX<0;所以我们认为θTX=0是一个决策边界，当它大于0或小于0时，逻辑回归模型分别预测不同的分类结果。
————————————————
原文链接：https://blog.csdn.net/program_developer/article/details/79113765
```

```
Logistic 回归模型是目前广泛使用的学习算法之一，通常用来解决二分类问题，虽然名字中有“回归”，但它是一个分类算法。
有些文献中译为“逻辑回归”，但中文“逻辑”与 logistic 和 logit 的含义相去甚远，因此下文中直接使用 logistic 表示。
Logistic 回归的优点是计算代价不高，容易理解和实现；缺点是容易欠拟合，分类精度可能不高。


现在是分成两类，如果数据需要分成三类或者更多该怎么办? 
 ---- 假如有A,B,C三类， 把其中A类做为1，BC做为0,然后做Logistic regression, 
得到模型a, 同理将B类做为1,AC作为0,得到模型b, 再同理得到模型c.    
最后测试的时候， 对任意一个数据点x, 我们能够得到x分别属于A,B,C三类的概率值 
最后比较大小，哪个大，这个x就属于哪一类
原文链接：https://blog.csdn.net/ppn029012/article/details/8775597
```

# 参考文档
通俗介绍神经网络中激活函数的作用
https://blog.csdn.net/program_developer/article/details/79190616
https://blog.csdn.net/program_developer/article/details/79113765
通俗介绍神经网络中激活函数的作用
https://blog.csdn.net/guanmao4322/article/details/93362157
https://github.com/Microstrong0305/machine_learning/commit/569c4d2ded0e8bd97c2cc7c9ea2c811868856b73
【机器学习】逻辑回归python实现
https://blog.csdn.net/huakai16/article/details/78140304

https://www.cnblogs.com/daniel-D/archive/2013/05/30/3109276.html

单变量线性回归模型
https://www.google.com/search?q=%E5%8D%95%E5%8F%98%E9%87%8F%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%E6%A8%A1%E5%9E%8B&oq=%E5%8D%95%E5%8F%98%E9%87%8F%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%E6%A8%A1%E5%9E%8B&aqs=chrome..69i57.25494j1j7&sourceid=chrome&ie=UTF-8


下面这几个讲的不错

通过介绍 单变量线性回归模型 简单的讲述了
机器学习的分类
假设函数
代价函数
梯度下降  全局最优解  局部最优解
下降算法中系数的下降函数
内容需要重点看
http://melonteam.com/posts/dan_bian_liang_xian_xing_hui_gui_mo_xing_jie_shao/

【机器学习】逻辑回归原理介绍　　讲的比较容易懂
https://blog.csdn.net/qq_24671941/article/details/94764693

对数公式
https://baike.baidu.com/item/%E5%AF%B9%E6%95%B0%E5%85%AC%E5%BC%8F

任意对数，反对数在线计算器
https://www.99cankao.com/algebra/logrithm.php

机器学习 --- 1. 线性回归与分类, 解决与区别    这个文章讲解了回归和分类的基本解决方法，简单容易理解 推荐
https://blog.csdn.net/ppn029012/article/details/8775597

交叉熵代价函数  中文翻译
https://hit-scir.gitbooks.io/neural-networks-and-deep-learning-zh_cn/content/chap3/c3s1.html
