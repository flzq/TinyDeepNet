# TinyDeepNet

TinyDeepNet 是一个轻量级的深度学习框架

## 技术栈
计算图、反向传播、自动求导机制、链式法则

## 依赖

核心代码
```
- python 3.7
- numpy
```

example
```
- pandas
- sklearn
```

## 主要模块
* 损失函数：CrossEntropyWithSoftMax
* 算子：Add，MatMul，SoftMax
* 优化器：GradientDescent，Momentum，AdaGrad

## 开始使用

运行示例代码
```
git clone git@github.com:flzq/TinyDeepNet.git

python example_iris.py
```

示例代码
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import TinyDeepNet as tdn


##################################
# 准备数据
##################################
# 读取鸢尾花数据集，去掉第一列Id
data = pd.read_csv("./Iris.csv").drop("Id", axis=1)

# 随机打乱样本顺序
data = data.sample(len(data), replace=False)

# 将字符串形式的类别标签转换成整数0，1，2
le = LabelEncoder()
number_label = le.fit_transform(data["Species"])

# 将整数形式的标签转换成One-Hot编码
oh = OneHotEncoder(sparse=False)
one_hot_label = oh.fit_transform(number_label.reshape(-1, 1))

# 特征列
features = data[['SepalLengthCm',
                 'SepalWidthCm',
                 'PetalLengthCm',
                 'PetalWidthCm']].values



##################################
# 构造计算图
##################################

# 构造计算图：输入向量，是一个4x1矩阵，不需要初始化，不参与训练
x = tdn.common.Variable(dim=(4, 1), init=False, trainable=False)

# One-Hot类别标签，是3x1矩阵，不需要初始化，不参与训练
one_hot = tdn.common.Variable(dim=(3, 1), init=False, trainable=False)

# 权值矩阵，是一个3x4矩阵，需要初始化，参与训练
W = tdn.common.Variable(dim=(3, 4), init=True, trainable=True)

# 偏置向量，是一个3x1矩阵，需要初始化，参与训练
b = tdn.common.Variable(dim=(3, 1), init=True, trainable=True)

# 线性部分
linear = tdn.ops.Add(tdn.ops.MatMul(W, x), b)

# 模型输出
predict = tdn.ops.SoftMax(linear)

# 交叉熵损失
loss = tdn.ops.loss.CrossEntropyWithSoftMax(linear, one_hot)



##################################
# 设置超参数
##################################

# 学习率
learning_rate = 0.02

# 构造Adam优化器
optimizer = tdn.optimizer.Momentum(tdn.default_graph, loss, learning_rate)

# 批大小为16
batch_size = 16



##################################
# 训练
##################################

# 训练执行200个epoch
for epoch in range(200):
    
    # 批计数器清零
    batch_count = 0
    
    # 遍历训练集中的样本
    for i in range(len(features)):
        
        # 取第i个样本，构造4x1矩阵对象
        feature = np.mat(features[i,:]).T
        
        # 取第i个样本的One-Hot标签，3x1矩阵
        label = np.mat(one_hot_label[i,:]).T
        
        # 将特征赋给x节点，将标签赋给one_hot节点
        x.set_value(feature)
        one_hot.set_value(label)
        
        # 调用优化器的one_step方法，执行一次前向传播和反向传播
        optimizer.one_step()
        
        # 批计数器加1
        batch_count += 1
        
        # 若批计数器大于等于批大小，则执行一次梯度下降更新，并清零计数器
        if batch_count >= batch_size:
            optimizer.update()
            batch_count = 0
            

    # 每个epoch结束后评估模型的正确率
    pred = []
    
    # 遍历训练集，计算当前模型对每个样本的预测值
    for i in range(len(features)):
                
        feature = np.mat(features[i,:]).T
        x.set_value(feature)
        
        # 在模型的predict节点上执行前向传播
        predict.forward()
        pred.append(predict.value.A.ravel())  # 模型的预测结果：3个概率值
    
    # 取最大概率对应的类别为预测类别
    pred = np.array(pred).argmax(axis=1)
    
    # 判断预测结果与样本标签相同的数量与训练集总数量之比，即模型预测的正确率
    accuracy = (number_label == pred).astype(np.int).sum() / len(data)
       
    # 打印当前epoch数和模型在训练集上的正确率
    print("epoch: {:d}, accuracy: {:.3f}".format(epoch + 1, accuracy))
```
