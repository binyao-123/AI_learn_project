# -*- coding: utf-8 -*-
# @Time:2025/9/29 17:17
# @Author:B.Yale
import torch
from d2l import torch as d2l
from torch import nn
from V3_3_SoftMax import train_ch3
from V3_3_SoftMax import predict_ch3

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
num_inputs, num_outputs,  = 784, 10
num_hiddens1 = 256
num_hiddens2 = 128

# 这种写法很傻，一层一层手写，有更好的办法
W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens1, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens1, requires_grad=True))

W2 = nn.Parameter(torch.randn(
    num_hiddens1, num_hiddens2, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_hiddens2, requires_grad=True))

W3 = nn.Parameter(torch.randn(
    num_hiddens2, num_outputs, requires_grad=True) * 0.01)
b3 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2, W3, b3]

def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

def net(X):
    X = X.reshape((-1, num_inputs))
    H1 = relu(X @ W1 + b1)
    H2 = relu(H1 @ W2 + b2)
    return H2 @ W3 + b3


loss = nn.CrossEntropyLoss(reduction='none')
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
predict_ch3(net, test_iter)
