import torch.nn as nn
import torch.nn.functional as F
import  torch.optim as optim
from torchvision import datasets,transforms
x.shape

#P40全连接层
#inherit from nn.Module
#init layer in _init_
#implement forward()
#autograd()和backward()是nn.Module会自动提供的

layer1=nn.Linear(784,200)
layer2=nn.Linear(200,200)
layer3=nn.Linear(200,10)

x=layer1(x)
x=F.relu(x,inplace=True)
#能用relu的地方尽可能用relu；但像是rgb的像素重建用sigmoid
x.shape

x=layer2(x)
x=F.relu(x,inplace=True)
x.shape

x=layer3(x)
x=F.relu(x,inplace=True)
x.shape

#step1
#这里就没有初始化问题;nn.Module有一套自己的初始化方法
class MLP(nn.Module):
    def __init__(self):#这里给自己的参数（例如hidden dimention），也可以不给
        super(MLP,self).__init__()
#step2
        self.model=nn.Sequential(
            nn.Linear(784,200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 10),
            nn.ReLU(inplace=True)
        )
#step3
    def forward(self,x):
        x=self.model(x)
        return x

#API风格介绍
#class-style API nn.Linear nn.Relu(不能访问内部的Tensor)
#function-style API F.relu F.cross-entropy

#Train
learning_rate=1e-3
epochs=5

device= torch.device('cuda:0')
net=MLP().to(device)
optimizer=nn.optim.SGD(net.parameters(),lr=learning_rate)
criteon = nn.CrossEntropyLoss()

for epoch in range(epochs):
    for batch_idx,(data,target) in enumerate(train_loader):
        data=data.view(-1,28*28)
        logits=net(data)
        loss=criteon(logits,target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


#激活函数
#循环神经网络里Tanh用得更多
#GPU加速
device=torch.device('cuda:0')
net=MLP().to(device)
optimizer=optim.SGD(net.parameters(),lr=learning_rate)
criteon=nn.CrossEntropyLoss().to(device)

for epoch in range(epochs):
    for batch_idx,(data,target) in enumerate(train_loader):
        data=data.view(-1,28*28)
        data,target=data.to(device),target.cuda()

#P42-49暂略