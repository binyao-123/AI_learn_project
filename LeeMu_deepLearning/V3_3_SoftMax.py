# -*- coding: utf-8 -*-
# @Time:2025/2/17 22:49
# @Author:B.Yale
import random

import torch
from IPython import display
from d2l import torch as d2l
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

'''
Softmax线性激活函数用于多分类问题，能够将模型的输出转换为概率分布。
主要特性包括：
1、将输入的未归一化数值转换映射为一个概率分布，概率总和为1，可以表示各类别的概率。
2、在训练阶段，Softmax函数与交叉熵损失函数结合使用，指导模型学习正确的概率分布。
3、在预测阶段，Softmax函数将模型输出转换为概率，选择概率最大的类别作为预测结果。
'''

'''
实现softmax函数三步骤：
1、对每个项求幂（exp）；
2、对每一行求和（小批量中每个样本是一行），得到每个样本的规范化常数；
3、将每一行除以其规范化常数，确保结果的和为 1。
'''

# softmax函数实现
def softmax(x):
    X_exp = torch.exp(x)
    # 对于矩阵来说，按每一行为单位做 softmax计算
    partition = X_exp.sum(1, keepdim=True)  # softmax本身并不压缩矩阵
    return X_exp / partition  # 广播机制


'''
交叉熵（Cross-Entropy）判断真实概率和预测概率之间差异，通常与softmax结合用作损失函数。（西瓜书 3.3节）
交叉熵是裁判，其核心目的：模型对正确答案预测的概率越高，损失就越小；概率越低，损失就越大。
通常用 torch.nn.CrossEntropyLoss 方法
'''
def cross_entropy(y_hat, y):
    # 交叉熵的精髓
    return - torch.log(y_hat[ range(len(y_hat)), y ])


def accuracy(y_hat, y):
    """统计预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)  # 返回指定维度最大值的索引张量
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter):
    """评估单个批次模型的精度"""
    if isinstance(net, torch.nn.Module):    # 检查确保传入的net是一个标准的PyTorch模型
        net.eval()  # 将模型设置为评估模式
    metric = d2l.Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():   # 关闭计算梯度，让评估过程更快
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_epoch_ch3(net, train_iter, loss, updater):
    """训练模型一个迭代周期"""
    if isinstance(net, torch.nn.Module):
        net.train()  # 将模型设置为训练模式
    # 训练损失总和、训练准确度总和、样本数
    metric = d2l.Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用 PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])

        if l.numel() > 1:
            # 直接对向量求和得到批次总损失(reduction='none' 的情况)
            batch_loss_sum = l.sum()
        else:
            # l 是一个标量 (reduction='mean' 的情况) 乘以批次大小来还原批次总损失
            batch_loss_sum = l * y.numel()

        metric.add(float(batch_loss_sum.item()), accuracy(y_hat, y), y.numel())

    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用 lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
        plt.draw()
        plt.pause(0.05)


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['train loss', 'train acc', 'test acc'])
    """训练整个模型"""
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    # 断言函数保护机制
    assert train_loss < 0.5, train_loss
    assert 1 >= train_acc > 0.7, train_acc
    assert 1 >= test_acc > 0.7, test_acc


def predict_ch3(net, test_iter, n=6):
    """预测标签"""
    for X, y in test_iter:
        break
    # 预测图的第一行（加粗）为真实标签，第二行预测标签
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    #titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    titles = [f'$\\bf{{{true}}}$\n{pred}' for true, pred in zip(trues, preds)]
    d2l.show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
    plt.show()


'''
# 定义softmax回归模型
def net(x):
    # torch.matmul：矩阵乘法函数
    # 在本例中，x是输入的样本，每一轮迭代共有 256个，即：
    # (256*784) * (784*10) -> (256*10)，即 256个样本分别在10个类别的预测概率
    return softmax(torch.matmul(x.reshape(-1, W.shape[0]), W) + b)

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)

batch_size = 256    # 数据迭代器的批量大小为256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 784    # 原始数据集图片像素 28*28，展平为 784的向量
num_outputs = 10    # 数据集有 10个类别

# 基于以上，权重构建为 784*10的矩阵，偏置构建为 1*10的行向量
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

lr = 0.1
num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
# train acc表示训练数据集上的精度，  test acc测试集上的精度
predict_ch3(net, test_iter)
'''
