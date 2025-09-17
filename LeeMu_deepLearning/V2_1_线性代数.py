# -*- coding: utf-8 -*-
# @Time:2025/2/11 23:35
# @Author:B.Yale
import torch

A = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
B = A.clone()  # 分配新内存，将A的副本分配给B，AB的 id不同
print(B)
print("****************************")

# 1 降维  可以指定特定维度求和，从而坍缩该维度。
# 对第一维求和（降维）后，A的形状由(2,3,4)变为(3,4)，被降维的数值广播到新张量
A_sum_dim0 = A.sum(dim=0)
# 同样，使用求均值函数 mean也可以降维
A_mean_dim0 = A.mean(dim=0)  # 等价于 (A.sum(axis=0) / A.shape[0])

# 2 非降维求和
# 计算保持轴数不变
print("原先A的形状：", A.shape)
sum_A = A.sum(dim=1, keepdim=True)
print("现在A的形状，被求和的维度保留为1：", sum_A.shape)
print("//////////////////////////////")

# cumsum：沿某个轴计算A元素的累积总和，加到另外的轴上
print(A)
print(A.cumsum(dim=0))
print("==================================")

# 3 求范数 norm
u = torch.tensor([3.0, -4.0])
print(torch.abs(u).sum())  # 计算 1范数
print(torch.norm(u))    # 针对向量的 2范数

print(torch.norm(torch.ones((4, 9))))   # 针对矩阵，计算一个 4×9 全 1 矩阵的 2 范数
