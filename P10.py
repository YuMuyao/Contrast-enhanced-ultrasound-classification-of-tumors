import numpy as np
import torch

#P10-P13

#创建tensor（关于numoy）

a=np.array([2,3.3])
torch.from_numpy(a)

a=np.ones([2,3])
torch.from_numpy(a)

#这里tensor函数（小写）接收的是具体的数据；Tensor/Float Tensor（）接收的是shape
torch.tensor([2,3.2])
torch.FloatTensor(2,3)

#申请一片控件（未初始化的数据填充；但数据奇怪
torch.empty(1)
torch.FloatTensor(2,2,2)#这里给的同样是shape，如果需要具体数据以list形式写入
torch.IntTensor(3,3,3,3)

#Tensor（）初始化类型是float，但使用中常用double；
torch.set_default_tensor_type(torch.DoubleTensor)
torch.tensor([1.2,3]).type()

#随机初始化
#rand是[0,1]的随机分布；randn是（0,1）的正态分布；normal的API操作更加复杂
a=torch.rand(3,3)#三行三列
torch.rand_like(a)
torch.normal(mean=torch.full([10],0),std=torch.arange(1,0,-0.1))
#使用arrange可以生成[start,end)的一个等差数列，默认1递增

#linsapce/logspace以steps参数值进行[start,end]区间的等长或者log长划分
torch.linspace(0,10,steps=4)
torch.logspace(0,10,steps=4)

#生成全1/全0/对角矩阵
#ones（）/zeros（）/eye（）
torch.ones(3,3)
torch.eye(3)

#randperm随机打散的种子，作用类似于random.shuffle

#(P12)Tensor的索引与切片
a=torch.rand(4,3,29,28)#4=batch_size;3=channel;28=height/weight
a[0]#第一张照片
a[0,0]#第一张图片的第一个通道








