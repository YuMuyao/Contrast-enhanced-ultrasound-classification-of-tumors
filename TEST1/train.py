from dataset import traindata
from network import ConvLSTM
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

if __name__=="__main__":
    t = 200
    h = 256
    w = 256
    c = 3
    lr=0.001
    weight_decay = 0e-5
    max_epoch = 1
    trainData = traindata(benignpath='D:/Study/超声造影/数据/华西数据/良性/', malignpath='D:/Study/超声造影/数据/华西数据/恶性/', t=200, h=256, w=256, c=3,batch_size=1)
    print("数据装载完成")
    model = ConvLSTM(input_size=(h, w),
                     input_dim=c,
                     hidden_dim=[64, 64, 32],
                     kernel_size=(3, 3),
                     num_layers=3,
                     batch_first=True,
                     bias=True,
                     return_all_layers=False)
    model=model.cuda()
    #print(model)
    # 目标函数及优化器
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(max_epoch):
        for ii, batch_sample in enumerate(trainData):
            # print(batch_sample['video'].shape,batch_sample['label'].shape)
            # print(len(batch_sample['label']))
            video=batch_sample['video'] #(b,t, c, h, w)
            label=batch_sample['label'] #(b,)
            b=len(label)
            video = torch.FloatTensor(video / 255.0)
            video, label = Variable(video).cuda(), Variable(label).cuda()  # .cuda()创建另一个变量，该变量不是计算图中的叶节点。使用它作为输入，所以它不会累积渐变
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            with torch.no_grad():
                score = model(video)
            loss = criterion(score, label)
            if ii % 80 == 0:
                print('epoch:', epoch, 'train_loss:', loss)
            loss.backward()
            optimizer.step()
        torch.save(model, 'model_' + str(epoch) + '.pkl')