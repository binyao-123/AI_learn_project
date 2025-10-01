# -*- coding: utf-8 -*-
# @Time:2025/2/23 19:01
# @Author:B.Yale
import torch
from d2l import torch as d2l
from torch import nn
from V3_3_SoftMax import train_ch3
from V3_3_SoftMax import predict_ch3

batch_size = 256    # 数据迭代器的批量大小为256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# 初始化单隐藏层的多层感知机，包含256个隐藏单元，两层线性变换
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]

# relu激活函数：小于 0 则置为 0，大于 0 则不变
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

# 模型
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X @ W1 + b1)
    return H @ W2 + b2


# 损失函数
loss = nn.CrossEntropyLoss(reduction='none')
# 训练
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
predict_ch3(net, test_iter)
