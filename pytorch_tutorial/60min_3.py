# NN
# nn.Module: a layer, a method forward(input),return output

# learnable parameters = weights
import torch
import torch.nn as nn
import torch.nn.functional as F

# torch.nn只接受mini-batch的输入，也就是说我们输入的时候是必须是好几张图片同时输入。
#
# 例如：nn. Conv2d 允许输入4维的Tensor：n个样本 x n个色彩频道 x 高度 x 宽度
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 使用super()继承时不用显式引用基类。

