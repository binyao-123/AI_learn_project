# -*- coding: utf-8 -*-
# @Time:2025/3/1 11:07
# @Author:B.Yale
import matplotlib.pyplot as plt
import torch
from d2l import torch as d2l
from V3_3_SoftMax import Animator

'''
正则化是处理过拟合的常用方法，一般可以使用L2正则化（即权重衰退）
权重衰退是一种防止模型过拟合的正则化方法。它的原理是在损失函数中增加一个与权重大小相关的惩罚项，
从而鼓励模型在训练过程中保持较小的权重值。

由于正则化项会对较大的权重给予更大的惩罚，梯度下降更新时不仅仅是朝着降低原始损失的方向调整参数，
同时也会让权重逐渐减小，这种现象就被称为“权重衰退”。
'''

# 创建数据集
# 训练数据越小，模型越复杂，越容易过拟合
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)

def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

def l2_penalty(w):
    """
    定义L2范数惩罚： w 的平方和的一半。
    平方项的系数2正好可以被抵消掉，使得导数形式更简洁，梯度就直接是权重本身，计算上非常优雅。
    """
    return torch.sum(w.pow(2)) / 2

def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 增加了L2范数惩罚项，广播机制使 l2_penalty(w)成为一个长度为 batch_size的向量
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数是：', torch.norm(w).item())


'''
lambd = 0 指禁用权重衰减后训练，此时训练误差减少，但测试误差没有减少（出现了严重的过拟合）
lambd > 0 使用权重衰减，w的L2范数明显开始下降
'''
train(lambd = 5)
plt.show()

