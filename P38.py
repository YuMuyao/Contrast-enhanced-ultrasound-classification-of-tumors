import torch
import torch.nn.functional as F

#交叉熵，熵越高越稳定，没有惊喜程度

a=torch.full([4],1/4.)
a*torch.log2(a)
-(a*torch.log2(a)).sum()
#输出的熵2.

a=torch.tensor([0.1,0.1,0.1,0.7])
-(a*torch.log2(a)).sum()
#输出的熵1.3568

a=torch.tensor([0.001,0.001,0.001,0.999])
-(a*torch.log2(a)).sum()
#输出的熵0.0313

#Numerical Stability
x=torch.randn(1,784)
w=torch.randn(10,784)

logits=x@w.t()
pred=F.softmax(logits,dim=1)

pred_log=torch.log(pred)

F.cross_entropy(logits,torch.tensor([3]))
#一个cross_entropy函数等同于softmax操作+log操作+nll_loss操作

F.nll_loss(pred_log,torch.tensor([3]))

 

