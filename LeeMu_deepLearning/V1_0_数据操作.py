# -*- coding: utf-8 -*-
# @Time:2025/2/6 21:57
# @Author:B.Yale
import torch

# 1.1 入门
# 张量（tensor）
x = torch.arange(12)
print(x)
print(x.shape)
print(x.numel())
print(torch.exp(x))     # exp(x) = e的 x次方
y = torch.zeros((2, 3, 4))      # 初始化 2个 3行 4列的张量，zeros全0，ones全1
print(y.shape)
print(y)
print("****************************")

# reshape可以重构当前张量的形状,参数-1为自动补全
# reshape(a,b)重构成二维张量        reshape(a,b,c)重构成三维张量 a个b行c列
# reshape相当于用新指定的结构来观察原张量，对 reshape的张量进行操作会影响到原张量
x = x.reshape(3, -1, 2)
print(x)
print(x.shape)
print("#############################")

# tensor创建一个具体的张量，randn创建随机张量，每个元素从均值为 0、标准差为 1的正态分布中采样
z = torch.tensor([[9, 9, 9, 9], [1, 2, 3, 4], [4, 3, 2, 1]])
z1 = torch.randn(3, 4)
print("-----------------------------")

# cat函数，特征融合，沿着批量维度向拼接，比如 dim=0时，两个shape为(3,4)的拼接后得到(6,4)
X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.arange(12, 24, dtype=torch.float32).reshape((3, 4))
print(torch.cat((X, Y), dim=0))
print(torch.cat((X, Y), dim=1))
print("//////////////////////////////")


# 1.2 广播机制
# 一个 a*b矩阵和 b*c矩阵，广播后得到 a*c矩阵
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(a)
print(b)
print(a+b)
print("==================================")

# 1.3 节省内存
# id类似 C语言的指针，重新对变量赋值，id也随之改变。
before = id(a)
a = a + b
print(id(a) == before)

# 保留 id的原地执行方法
Z = torch.zeros_like(Y)     # torch.zeros_like函数生成相同形状的全 0张量
print('old id(Z):', id(Z))
Z[:] = X + Y    # 切片赋值表示对 c 的所有元素进行赋值操作
print('new id(Z):', id(Z))

# 也可以对该张量直接进行赋值操作，保留id
before = id(X)
X += Y
print(id(X) == before)
