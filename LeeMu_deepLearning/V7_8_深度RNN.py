# -*- coding: utf-8 -*-
# @Time:2025/11/13 16:57
# @Author:Binyao

import torch
from torch import nn
from d2l import torch as d2l
from V7_3_NLP import load_data_time_machine
from V7_4_RNN import train_ch8
from V7_5_RNN_Simple import RNNModel

'''
在深度循环神经网络中，隐状态的信息被传递到当前层的下一时间步和下一层的当前时间步。

有许多不同风格的深度循环神经网络， 如长短期记忆网络、门控循环单元、或经典循环神经网络。 这些模型在深度学习框架的高级API中都有涵盖。

总体而言，深度循环神经网络需要大量的调参（如学习率和修剪） 来确保合适的收敛，模型的初始化也需要谨慎。
'''

batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
device = d2l.try_gpu()
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
model = RNNModel(lstm_layer, len(vocab))
model = model.to(device)
num_epochs, lr = 500, 2
train_ch8(model, train_iter, vocab, lr*1.0, num_epochs, device)
