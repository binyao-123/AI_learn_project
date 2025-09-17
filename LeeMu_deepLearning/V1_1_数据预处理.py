# -*- coding: utf-8 -*-
# @Time:2025/2/11 22:35
# @Author:B.Yale
import os
import pandas as pd
import torch

# 存储在目录   ../data/house_tiny.csv
os.makedirs(os.path.join('../../PythonProject', 'data'), exist_ok=True)
data_file = os.path.join('../../PythonProject', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Name,Price\n')  # 列名
    f.write('NA,Bin,127500\n')  # 每行表示一个数据样本
    f.write('2,Yale,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,NA\n')
data = pd.read_csv(data_file)
print(data)

# 处理缺失的数据包括插值法和删除法。插值法使用 fillna 函数
inputs = data.iloc[:, :]
inputs = inputs.fillna(inputs.mean(numeric_only=True))  # 取缺失数字列的平均值
outputs = inputs.iloc[:, 2]
print("==============数字列已完成填充====================")
print("inputs:\n", inputs)
print("outputs:\n", outputs)
print("================================================")

# 在上面数据集中，Name这一列有 Bin、Yale字符串，其他为空（NA）
# get_dummies：为这些字符串创建Alley_xxx，有 xxx 的记为 1，其他记为 0
# dummy_na=True：增加一列“Name_nan”
# dtype=int：将布尔类型表示为1/0.
inputs = pd.get_dummies(inputs, dummy_na=True, dtype=int)
print("字符类型:\n", inputs)

# 现在 input、output都是纯数值了，可以转化为张量
X = torch.tensor(inputs.values)
y = torch.tensor(outputs.values)
print(X)
print(y)
