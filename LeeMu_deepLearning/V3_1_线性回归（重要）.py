# -*- coding: utf-8 -*-
# @Time:2025/2/16 23:41
# @Author:B.Yale
import torch
from torch import nn  # Neural Network
from torch.utils import data
from d2l import torch as d2l
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

# 使用 d2l库模拟生成 1000个样本点的线性回归数据集，true_w权重，true_b偏差
true_w = torch.tensor([3, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
# 显示数据集
plt.figure(figsize=(6, 5))
plt.xlabel("Feature value")
plt.ylabel("Label")
plt.axhline(y=0, color='black', linewidth=0.5, linestyle='-.')
plt.axvline(x=0, color='black', linewidth=0.5, linestyle='--')
plt.scatter(features[:, 0], labels, 1)  # 展示第一个特征和标签之间的线性关系
plt.scatter(features[:, 1], labels, 1)  # 展示第二个特征和标签之间的线性关系
plt.show()

'''**** TensorDataset + dataLoader 为推荐写法 **** '''
# 读取数据，将特征值和标签值作为参数传递，指定 batch_Size
# shuffle=is_train表示数据迭代器在每个迭代周期内打乱数据
def load_array(data_arrays, batch_size, is_train=True):
    # 构造一个PyTorch数据迭代器
    # *是一个解包（unpacking）操作符。将一个列表（list）或元组（tuple）中的所有元素解开，并将它们作为独立参数传递给函数。
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_Size = 10
data_iter = load_array((features, labels), batch_Size)  # 10个一批次打包数据集
print(next(iter(data_iter)))  # 打印第一批次
print("***********************************************")

# 定义模型 net，实现单层的神经网络（由于没有隐含层，因此不需要使用 Relu等激活函数）
net = nn.Sequential(nn.Linear(2, 1))   # 输入的特征(权重)维度为 2，输出维度 1的特征
print("net打印：", net)
# net[0]网络中第一个图层， 使用 weight.data和 bias.data方法访问参数
print(net[0].weight.data)
print(net[0].bias.data)
# 初始化参数（先搭建好框架并获得默认参数，然后为了训练的稳定性、可复现性和有效性，再手动将这些参数重置为我们想要的特定初始值）
# normal_和 fill_重置参数值, normal_(a,b)表示均值 a，标准差 b的正态分布值
print("初始化权重:", net[0].weight.data.normal_(0, 0.01))
print("初始化偏差:", net[0].bias.data.fill_(0))
### 定义损失函数 loss
loss = nn.MSELoss()     # 均方误差(Mean Squared Error, 也是L2范数) ，返回样本损失的平均值
### 定义优化算法，SGD为小批量随机梯度下降算法，学习率 lr为 0.03
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

'''**** 训练迭代推荐写法 **** '''
num_epochs = 3
print("可以看到损失已经逐步缩小：")
for epoch in range(num_epochs):
    for X, y in iter(data_iter):
        L = loss(net(X), y)     # 计算特征和标签之间的方差
        trainer.zero_grad()     # 梯度清零
        L.backward()        # 求导
        trainer.step()      # 更新模型

    L = loss(net(features), labels)     # print展示当前数据集训练效果
    print(f'epoch {epoch + 1}, loss {L:f}')

print("神经网络初始化的两个参数：权重和偏差已经非常接近模型值预设的w=[2, -3.4] 和 b=4.2")
print(net[0].weight.data)
print(net[0].bias.data)
