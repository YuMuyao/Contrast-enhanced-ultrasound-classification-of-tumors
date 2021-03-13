import torch
import torch.nn as nn
import torch.functional as F
#P58经典的卷及网络
#Res网络和Inception推荐;VGG太老了
#LeNet
#GoogleNet
#ResNet
class ResBlk(nn.Module):
    def __init__(self,ch_in,ch_out):
        self.conv1=nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1)
        self.bn1=nn.BatchNorm2d(ch_out)
        self.conv2=nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1)
        self.bn2=nn.BatchNorm2d(ch_out)

        self.extra=nn.Sequential()
        if ch_out!=ch_in:
            self.extra=nn.Sequential(
                nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1),
                nn.BatchNorm2d(ch_out)
            )

        def forward(self,x):
            out=F.relu(self.bn1(self.conv1(x)))
            out=self.bn2(self.conv2(out))
            out=self.extra(x)+out
            return out