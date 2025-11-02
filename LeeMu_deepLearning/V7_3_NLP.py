# -*- coding: utf-8 -*-
# @Time:2025/10/31 17:54
# @Author:Binyao

import random
import torch
from d2l import torch as d2l
import re
import collections


'''
语言模型是自然语言处理的关键。

n元语法通过截断相关性，为处理长序列提供了一种实用的模型。

长序列存在一个问题：它们很少出现或者从不出现。

齐普夫定律支配着单词的分布，这个分布不仅适用于一元语法，还适用于其他
元语法。

通过拉普拉斯平滑法可以有效地处理结构丰富而频率不足的低频词词组。

读取长序列的主要方式是随机采样和顺序分区。在迭代过程中，后者可以保证来自两个相邻的小批量中的子序列在原始序列上也是相邻的。
'''
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():  #@save
    """将时间机器数据集加载到文本行的列表中"""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    # 使用正则表达式替换非字母字符为空格，并转为小写
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

'''
lines = read_time_machine()
tokens = d2l.tokenize(lines)
corpus = [token for line in tokens for token in line]
vocab = d2l.Vocab(corpus)

freqs = [freq for token, freq in vocab.token_freqs]
bigram_tokens = [' '.join(pair) for pair in zip(corpus[:-1], corpus[1:])]
bigram_vocab = d2l.Vocab(bigram_tokens)

trigram_tokens = [' '.join(triple) for triple in zip(
    corpus[:-2], corpus[1:-1], corpus[2:])]
trigram_vocab = d2l.Vocab(trigram_tokens)

bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
corpus_indices = [vocab[token] for token in corpus]
'''

def seq_data_iter_random(corpus, batch_size, num_steps):  #@save
    """使用随机抽样生成一个小批量子序列"""
    # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 减去1，是因为我们需要考虑标签
    num_subseqs = (len(corpus) - 1) // num_steps
    # 长度为num_steps的子序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 在随机抽样的迭代过程中，
    # 来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
    random.shuffle(initial_indices)

    def data(pos):
        # 返回从pos位置开始的长度为num_steps的序列
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，initial_indices包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)


def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
    """使用顺序分区生成一个小批量子序列"""
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y


class SeqDataLoader:  # @save
    """加载序列数据的迭代器"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random  # 使用我们自己定义的函数
        else:
            self.data_iter_fn = seq_data_iter_sequential  # 使用我们自己定义的函数

        # 使用我们自己的逻辑加载数据
        lines = read_time_machine()
        tokens = d2l.tokenize(lines)
        corpus_tokens = [token for line in tokens for token in line]
        if max_tokens > 0:
            corpus_tokens = corpus_tokens[:max_tokens]
        self.vocab = d2l.Vocab(corpus_tokens)
        self.corpus = [self.vocab[token] for token in corpus_tokens]
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_data_time_machine(batch_size, num_steps,  #@save
                           use_random_iter=False, max_tokens=2000):
    """返回时光机器数据集的迭代器和词表"""
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab


