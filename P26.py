import numpy as np
import torch.nn.functional as F
import torch

#sigmoid(可能会带来梯度离散和梯度爆炸)
a=torch.linspace(-100,100,10)
torch.sigmoid(a)
F.sigmoid(a)

#tanh
a=torch.linspace(-1,1,10)
torch.tanh(a)

#relu(rectified linear unit)
a=torch.linspace(-1,1,10)
torch.relu(a)
F.relu(a)

#Typical loss
#MSE(均方差)Mean Squared Error(实际值减去模型输出值的平方和)
#Cross Entr opy Loss(交叉熵)：二分类、多分类

#autograd,grad（torch.autograd.grad(loss,[w1,w2,...]),返回[w1 grad,w2 grad...]）
#需要告诉grad哪些变量是需要gradient的
x=torch.ones(1)
w=torch.full([1],2)
mse=F.mse_loss(torch.ones(1),x*w)#作为动态图的建图

torch.autograd.grad(mse,[w])#此时无法对w进行求导（因为w已经给出）

w.requires_grad_()#这时候告诉grad参数w是可以求导的
#或者采用w=torch.len([1],requiregrd=True)

mse=F.mse_loss(torch.ones(1),x*w)
torch.autograd.grad(w)

#loss.backward()(不直接返回list，可以通过w1.grad\w2.grad查看)

#softnax函数（所有值都在（0,1）之间，且和为1）
a=torch.rand(3)
a.requires_grad_()
p=F.softmax(a,dim=0)#p是返回的概率，dim指在那个维度上操作

p.backward()#这里会报错，因为只能backward一张图，计算之后会被清除

p=F.softmax(a,dim=0)
torch.autograd.grad(p[1],[a],retain_graph=True)

torch.autograd.grad(p[2],[a])

#P32单层感知机
x=torch.randn(1,10)
w=torch.randn(1,10,requires_grad=True)

o=torch.sigmoid(x@w.t())
o.shape

loss=F.mse_loss(torch.ones(1,1),o)
loss.shape

loss.backward()

w.grad

#单输出感知机-->多输出感知机
x=torch.randn(1,10)
w=torch.randn(2,10,requires_grad=True)

o=torch.sigmoid(x@w.t())
o.shape

loss=F.mse_loss(torch.ones(1,2 ),o)
loss

loss.backward()

w.grad

#链式法则
x=torch.tensor(1.)
w1=torch.tensor(2.,requires_grad=True)
b1=torch.tensor(1.)
w2=torch.tensor(2.,requires_grad=True)
b2=torch.tensor(1.)

y1=x*w1+b1
y2=y1*w2+b2

dy2_dy1=torch.autograd.grad(y2,[y1],retain_graph=True)[0]

dy1_dw1=torch.autograd.grad(y1,[w1],retain_graph=True)[0]

dy2_dw1=torch.autograd.grad(y2,[w1],retain_graph=True)[0]

dy2_dy1*dy1_dw1

dy2_dw1

#多层感知机（MLP）的反向传播

#D函数优化实例
