import torch
import torch.nn as nn

#P57 batch_normalize的好处
#converge faster收敛更快
#better performance
#robust更稳定(stable/larger learning rate)

x=torch.rand(100,16,784)
layer=nn.BatchNorm1d(16)
out=layer(x)

layer.running_mean#统计出的总的μ(全局)

layer.running_var#统计出的总的δ(全局)
#输入在batch上的x值（需要学习的值是γ和β，在反向传播的时候更新）
#计算出μ均值
#计算δ^2方差
#x'=(x-μ)/√(δ²+ε)
#y'=γ*x'+β

#batch normalize对2d数据的操作
x.shape#1,16,7,7
layer=nn.BatchNorm2d(16)#传入channel的数据

out=layer(x)

layer.weight#w-->γ
layer.bias#b-->β

vars(layer)

#affine γ和δ会不会自动学习

#在train和test的时候batch_normalize是不同的
#只有一个sample
#μ和δ²<--running_mean、running_var
#不需要γ和β

#test部分
layer.eval()
#out:BatchNorm1d(16,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)

out=layer(x)