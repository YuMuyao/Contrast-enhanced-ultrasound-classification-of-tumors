import os
import numpy as np
import torch
import torch.utils.data as Data


def traindata(benignpath='D:/Study/超声造影/数据/华西数据/良性/', malignpath='D:/Study/超声造影/数据/华西数据/恶性/', t=200, h=256, w=256, c=3, batch_size=10):
    benigndir = os.listdir(benignpath)
    maligndir = os.listdir(malignpath)
    train = []
    label = []
    for i in benigndir:
        with open(str(benignpath + i), 'rb') as f:
            X = np.frombuffer(f.read(), dtype=np.uint8)
            X = np.reshape(X, (t, c, h, w))
            train.append({'video': X, 'label': 0})
    print('良性', train.__len__())
    for i in maligndir:
        with open(str(malignpath + i), 'rb') as f:
            X = np.frombuffer(f.read(), dtype=np.uint8)
            X = np.reshape(X, (t, c, h, w))
            train.append({'video': X, 'label': 1})
    print('总共', train.__len__())
    print(train.__len__())

    ###
    train_dataset = Data.DataLoader(
        dataset=train,  # 打包tensor
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # RAM越大可以设置越大，加载每个batch更快
    )
    return train_dataset
