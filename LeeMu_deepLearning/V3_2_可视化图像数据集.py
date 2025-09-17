# -*- coding: utf-8 -*-
# @Time:2025/2/17 19:33
# @Author:B.Yale
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
d2l.use_svg_display()

# ToTensor函数：将图像数据从 PIL类型变换成 32位浮点数格式，除以 255，使所有像素的数值在 0～1之间
# 下载由 10个类别的图像组成，每个类别由训练数据集中的60000张图像和测试数据集中的10000张图像组成
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)
print(mnist_train[0][0].shape)  # 输出[1, 28, 28]，1表示灰色，28*28像素
print("********************************************")


def get_fashion_mnist_labels(labels):
    """返回 Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


# 读取小批量
get_dataloader_workers = 4
# def get_dataloader_workers():
#     """使用4个进程来读取数据"""
#     return 4


# 可视化样本
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
plt.show()

batch_size = 256
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_workers)
'''
下面代码的目的就是把“数据读取”和“模型计算”这两件事分开，单独测量前者需要多久，帮你判断数据加载是不是太慢了。
如果这个时间很长，你就需要考虑优化你的数据加载流程了。
'''
timer = d2l.Timer()
for X, y in train_iter:
    continue
print(f'{timer.stop():.2f} sec')  # 读取训练数据所需的时间

'''
# 以后可以调用事先封装好的函数 load_data_fashion_mnist代替上述操作
# d2l库中该函数定义如下：
def load_data_fashion_mnist(batch_size, resize=None):
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))


train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break
'''
