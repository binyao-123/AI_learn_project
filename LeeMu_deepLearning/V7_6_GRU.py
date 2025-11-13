# -*- coding: utf-8 -*-
# @Time:2025/11/2 14:50
# @Author:Binyao
import torch
from d2l import torch as d2l
from V7_3_NLP import load_data_time_machine
from V7_5_RNN_Simple import RNNModel
from V7_4_RNN import train_ch8
from torch import nn
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

'''
门控循环神经网络可以更好地捕获时间步距离很长的序列上的依赖关系。

重置门(reset gate)决定在计算“候选隐藏状态”(h̃_t)时，要“忽略”多少过去的隐藏状态(h_{t-1})，适合捕获序列中的短期依赖关系。

更新门(update gate)有助于捕获序列中的长期依赖关系。能够把关键变化的信息记录进隐状态。让重要的信息在长序列中无损地传递。

有点像把传统的RNN正则化，即抛弃一些模型认为没什么用的信息。两个门相互控制开关的大小，调配当前输入和过去输入对隐状态的影响


公式：h_t = z_t ⊙ h_{t-1} + (1 - z_t) ⊙ h̃_t

场景	    重置门 r_t	    更新门 z_t	    技术效果	                                    核心作用
1	    ≈ 1 (不重置)	    ≈ 0 (全更新)	    h_t 主要由x_t和h_{t-1}计算的新状态决定	        退化为标准RNN
2	    任意	            ≈ 1 (不更新)	    h_t 直接复制 h_{t-1}，忽略 x_t	            捕获长期依赖 (信息直通)
3	    ≈ 0 (全重置)	    任意 (通常>0)	h̃_t 只由x_t决定，h_{t-1}的影响被重置	        捕获短期依赖 (开启新上下文)
'''



def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    W_xz, W_hz, b_z = three()  # 更新门参数
    W_xr, W_hr, b_r = three()  # 重置门参数
    W_xh, W_hh, b_h = three()  # 候选隐状态参数
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )


def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)



batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
num_inputs = vocab_size
gru_layer = nn.GRU(num_inputs, num_hiddens)
model = RNNModel(gru_layer, len(vocab))
model = model.to(device)
train_ch8(model, train_iter, vocab, lr, num_epochs, device)
plt.show()

