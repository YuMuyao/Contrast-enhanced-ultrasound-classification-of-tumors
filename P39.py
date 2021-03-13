import torch
import torch.nn.functional as F

#第一版梯度长时间得不到更新，产生了梯度离散
# torchvision 包收录了若干重要的公开数据集、网络模型和计算机视觉中的常用图像变换
import torchvision
import torchvision.transforms as transforms

#多分类实战minist

batch_size=200
learning_rate=0.01
epochs=10

train_loader=torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(root='./mnist',train=True,download=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,),(0.3081,))
                   ])),
    batch_size=batch_size,shuffle=True)

test_loader=torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(root='./mnist',train=False,
                   transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1327,),(0.3081,))
                    ])),
    batch_size=batch_size,shuffle=True)

w1,b1=torch.randn(200,784,requires_grad=True),\
    torch.zeros(200,requires_grad=True)
w2,b2=torch.randn(200,200,requires_grad=True),\
    torch.zeros(200,requires_grad=True)
w3,b3=torch.randn(10,200,requires_grad=True),\
    torch.zeros(10,requires_grad=True)

#初始化非常重要，这里的b1、b2、b3已经全部初始化为0；虽然之前已对w1、w2、w3进行高斯初始化，但效果并不尽人意；
torch.nn.init.kaiming_normal(w1)
torch.nn.init.kaiming_normal(w2)
torch.nn.init.kaiming_normal(w3)

def forward(x):
    x=x@w1.t()+b1
    x=F.relu(x)
    x=x@w2.t()+b2
    x=F.relu(x)
    x=x@w3.t()+b3
    x=F.relu(x)
    return x
#这里返回的是logits(没有经过softmax或者是sigmoid)

optimizer=torch.optim.SGD([w1,b2,w2,b3,w3,b3],lr=learning_rate)
criteon = torch.nn.CrossEntropyLoss()

for epoch in range(epochs):
    for batch_idx,(data,target) in enumerate(train_loader):
        data=data.view(-1,28*28)

        logits=forward(data)
        loss=criteon(logits,target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx%100==0:
            print('Train Epoch:{} [{}/{} ({:.0f}%) ]\tLoss:{:.6f}'.format(
                epoch,batch_idx*len(data),len(train_loader.dataset),
                100.*batch_idx/len(train_loader),loss.item()))

    test_loss=0
    correct=0
    for data,target in test_loader:
        data=data.view(-1,28*28)
        logits=forward(data)
        test_loss+=criteon(logits,target).item()

        pred=logits.data.max(1)[1]
        correct+=pred.eq(target.data).sum()

    test_loss/=len(test_loader.dataset)
    print('\nTest_set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss,correct,len(test_loader.dataset),
        100.*correct/len(test_loader.dataset)))


