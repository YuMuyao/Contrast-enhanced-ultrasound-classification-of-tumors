import os

import scipy.io as scio
import numpy as np
import transplant
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from sklearn.metrics import auc

if __name__ == "__main__":
    #matlab = transplant.Matlab(jvm=False, desktop=False)#调用matlab
    ###读取良性肿瘤数据
    dataFile = 'D://Study//超声造影//数据//B.mat'
    data = scio.loadmat(dataFile)["video"]  # (771, 648, 548, 3)
    X = data.reshape((771, 3, 648, 548))
    #X = X / 255  # 像素范围0-255故除以255规范到0-1

    ###计算每帧每个像素的强度值（邻域平均）
    tic = np.zeros((700, 648, 548))  # 存储每帧每个像素点最终强度值，格式为771帧*图像大小648*548
    ticmap = np.zeros((700, 648, 548)) #存储每个像素的tic曲线
    ##做法一：对RGB三个通道分别进行二维卷积（对每帧每个通道进行3*3卷积，结果/9取八邻域平均值，/255像素值归一化，再乘上RGB比重，最后三通道结果相加）
    k = np.ones((3, 3))

    for i in range(0,700):
        #原论文还对每帧图像进行了高斯滤波
        #/////////////////////////////高斯滤波，待填坑///////////////////////////////////////////
        XX=X[i][0]* 0.3 / 255+X[i][1]* 0.59 / 255+X[i][2]* 0.11 / 255
        tic[i] = signal.convolve2d(XX, k/9, mode='same')

    #遍历每个像素点，进行sg滤波拟合和求取特征值
    pi = np.zeros((648, 548))  # 存储每个像素的PI
    ttp = np.zeros((648, 548))  # 存储每个像素的TTP
    for x in range(0,648):
        for y in range(0,548):
            print(tic[:,x,y].shape)
            ticmap[:,x,y] = savgol_filter(tic[:,x,y], 31, 2, mode='nearest') #s-g滤波拟合
            pi[x][y]=np.max(ticmap[:,x,y], axis=0) #求PI
            ttp[x][y] = np.argmax(ticmap[:,x,y], axis=0)  # 求TTP


    #####方法二：一次性全图sg滤波
    #ticmap=savgol_filter(tic, 31, 2, mode='nearest')
    ##ticmap = matlab.smooth(tic, 35, 'rloess')
    #print(ticmap.shape)

    #保存计算好的造影强度数据
    # mdic = {"i": tic, "label": "intensity"}
    # scio.savemat("D://output/B_intensity.mat",mdic)
    mdic = {"i": tic, "label": "intensity"}
    scio.savemat("D://output/B_tic.mat",mdic)
    mdic1 = {"pi": pi, "label": "peak"}
    mdic2 = {"ttp": ttp, "label": "time"}
    scio.savemat("D://output/B_pi.mat", mdic1)
    scio.savemat("D://output/B_ttp.mat", mdic2)