# -*- coding: utf-8 -*-
# @Time:2025/9/29 18:19
# @Author:B.Yale
import torch
from torch import nn
from d2l import torch as d2l
from V3_3_SoftMax import train_ch3
from V3_3_SoftMax import predict_ch3

# 构建一个拥有10个隐藏层的深度网络
hidden_layers = []
in_features = 784

for i in range(10):
    out_features = 64 # 假设每层有64个神经元
    hidden_layers.append(nn.Linear(in_features, out_features))
    hidden_layers.append(nn.BatchNorm1d(out_features)) # 1、训练过程中的每一次前向传播进行批量归一化
    hidden_layers.append(nn.ReLU())
    in_features = out_features # 下一层的输入就是这一层的输出

net_deep = nn.Sequential(nn.Flatten(),
                         *hidden_layers,  # 解包，将列表中的所有层展开并传入
                         nn.Linear(in_features, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu') # 2、引用何恺明方法初始化Relu函数
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


net_deep.apply(init_weights)
batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss(reduction='mean')
trainer = torch.optim.SGD(net_deep.parameters(), lr=lr)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
train_ch3(net_deep, train_iter, test_iter, loss, num_epochs, trainer)
predict_ch3(net_deep, test_iter)
