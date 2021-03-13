import torch
import torch.functional as F
import torch.nn as nn
from torchvision import transforms,datasets

#P50卷积
#一层：指的是该层的权值和输出
layer=nn.Conv2d(1,3,kernel_size=3,stride=1,padding=0)
#1指的是input的channel,3指的是kernel的数量
x=torch.rand(1,1,28,28)

#filter/kernel/weight指的是一个概念

out=layer.forward(x)

layer=nn.Conv2d(1,3,kernel_size=3,stride=1,padding=1)
out=layer.forward(x)

layer=nn.Conv2d(1,3,kernel_size=3,stride=2,padding=1)
#stride在某些情况下带有降维的功能
out=layer.forward(x)

out=layer(x)#推荐方法(调用__call__ ；有pytorch自带的hooks函数)(会先运行实例再运行forward函数；同时完成一些pytorch自带的功能建设 )

#inner weight&bias
layer.weight.shape#torch.Size([3,1,3,3])

layer.bias.shape#torch.Size([3])

w=torch.rand(16,3,5,5)
b=torch.rand(16)

out=F.conv2d(x,w,b,stride=1,padding=1)

x=torch.randn(1,3,28,28)
out=F.conv2d(x,w,b,stride=1,padding=1)

#P55 pooling下采样-->max/avg pooling
#upsample上采样

x=out

layer=nn.MaxPool2d(2,stride=2)#window的大小和补偿的大小

out=layer(x)

out=F.avg_pool2d(x,2,stride=2)

#F.interpolate
x=out
out=F.interpolate(x,scale_factor=2,mode='nearest')#scale_factor是放大倍数，mode是紧邻差值模式
out.shape

out=F.interpolaye(x,scale_factor=3,mode='nearest')

out.shape

#卷积函数常用单元Unit:conv2d-->batch normalization-->pool-->relu(顺序可以颠倒)
x.shape

layer=nn.ReLU(inplace=True)#设置inplace为true的结果是，x'会占据x原本的空间（正常情况下会节省一半的空间）
out=layer(x)
out.shape

out=F.relu(x)
out.shape

#batch-normalization-->(某些时候必须使用sigmoid)我们需要把输入控制在一定范围内
#权值缩放的概念
normalize=transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
#batch norm/layer norm/instance norm/group norm
#batch norm指的是计算同一个channel的不同实例
