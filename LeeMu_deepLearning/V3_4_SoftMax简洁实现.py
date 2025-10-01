# -*- coding: utf-8 -*-
# @Time:2025/2/20 0:10
# @Author:B.Yale
import torch
from torch import nn
from d2l import torch as d2l
from V3_3_SoftMax import train_ch3
from V3_3_SoftMax import predict_ch3

batch_size = 256    # 数据迭代器的批量大小为256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# Flatten展平层，在线性层前调整网络输入的形状（将28 * 28图像压缩成784）
# Linear线性层，输入展平层的784，输出数据集中的 10 个类别，output = input @ W^T + b
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    """初始化权重、偏置"""
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
        nn.init.zeros_(m.bias)


net.apply(init_weights) # init_weights 函数会去遍历访问 net 配的参数 m，即从Flatten->Linear。

# 定义损失函数
loss = nn.CrossEntropyLoss(reduction='none')    # 隐式地实现了 softmax + 交叉熵操作，经典用法
# 定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

# 训练
num_epochs = 5
train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
predict_ch3(net, test_iter)
