# -*- coding: utf-8 -*-
# @Time:2025/3/5 15:45
# @Author:B.Yale
import matplotlib.pyplot as plt
import torch
from torch import nn
from d2l import torch as d2l
from V3_3_SoftMax import train_ch3
from V3_3_SoftMax import predict_ch3

'''
1、
dropout在前向传播过程中，计算每一内部层的同时将某些神经元的输出置为0，是为了减少神经网络的过拟合。
从表面上看是在训练过程中丢弃（drop out）一些神经元。
只在训练时丢弃，预测时仍然要使用所有权重。
2、
在整个训练过程的每一次迭代中，标准dropout在计算下一层之前将当前层中的一些节点置零。
3、
dropout和L2都是正则项。正则项只在训练的时候使用，主要对权重产生影响。
加入正则的意义是在更新权重时让模型复杂度更低一些
进行预测的时候模型已经训练完毕了（权重已经固定）
4、
可以将高斯噪声添加到线性模型的输入中。
'''

'''
不稳定梯度带来的风险不止在于数值表示，也威胁到优化算法的稳定性。
梯度爆炸（gradient exploding）问题： 参数更新过大，破坏了模型的稳定收敛； 
梯度消失（gradient vanishing）问题：参数更新过小，在每次更新时几乎不会移动，无法学习。
因此，数值稳定的目标是让梯度值在合理的单位内。
具体方法实现：
    1、将乘法变加法：LSTM、RseNet
    2、梯度归一化、梯度裁剪
'''

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # dropout为 1 时，所有元素都被丢弃
    if dropout == 1:
        return torch.zeros_like(X)
    # dropout为 0 时，所有元素都被保留
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()
    # 除以(1.0 - dropout) 是为了缩放补偿，因为丢弃了一些节点，不补偿会导致训练和预测时的输出尺度不一致
    return mask * X / (1.0 - dropout)


num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 2048, 2048
# 分别设置第一个、第二个隐藏层的dropout概率（每个神经元有 p 的概率被置为零）
# dropout1, dropout2 = 0, 0
dropout1, dropout2 = 0.5, 0.5


class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training=True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # 只有在训练模型时才使用dropout
        if self.training:
            # 在第一个全连接层之后添加一个dropout层
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training:
            # 在第二个全连接层之后添加一个dropout层
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out


net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
num_epochs, lr, batch_size = 20, 0.001, 256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 为了体现dropout作用，需要在dropout置为0的时候模型产生过拟合现象，因此只选取数据集前5000个，改用Adam优化器
subset_indices = list(range(5000))
train_subset = torch.utils.data.Subset(train_iter.dataset, subset_indices)
train_iter_small = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)

trainer = torch.optim.Adam(net.parameters(), lr=lr)
train_ch3(net, train_iter_small, test_iter, loss, num_epochs, trainer)
# predict_ch3(net, test_iter)
plt.show()

