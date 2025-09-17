# -*- coding: utf-8 -*-
# @Time:2025/2/16 22:23
# @Author:B.Yale
import torch

x = torch.arange(4.0)   # tensor([0., 1., 2., 3.])
# 存储梯度
x.requires_grad_(True)  # 等价于x=torch.arange(4.0,requires_grad=True)
print(x.grad)   # 默认值是None
# 对函数 y = 2 * (x^T*x) 的列向量求导
y = 2 * torch.dot(x, x)
# backward函数：自动计算 y关于 x每个分量的梯度：y’ = 4x
# gard函数：保存 x 的导数（梯度）结果
y.backward()    # 求导
print(x.grad)   # 得到梯度值
print("****************************")

# 另一个例子
x.grad.zero_()  # PyTorch默认累积梯度，需要清除之前的值
y = x.sum()     # 即 x = (x_1 + x_2 + ... + x_n)
y.backward()    # y' = 1
print(x.grad)
print("#############################")

# 非标量变量的反向传播。
x.grad.zero_()
y = x * x   # 等价于y.backward(torch.ones(len(x)))
y.sum().backward()   # y先求和转换成一个标量，然后再对标量求导，y' = 2x
print(x.grad)
print("-------------------------------------")

# 分离计算
x.grad.zero_()
y = x * x
u = y.detach()      # 用 u代替 y，将 y从函数关系中抽离成常量
print("u：", u)
z = u * x           # z = 常量*x，对x求导，即z' = u。
z.sum().backward()
print("x.grad：", x.grad)
print(x.grad == u)
