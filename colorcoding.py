import scipy.io as scio
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.signal import savgol_filter, find_peaks
from sklearn.metrics import auc
import matplotlib.image as mp
if __name__ == "__main__":
    dataFile = 'D://output//B_pi.mat'
    pi = scio.loadmat(dataFile)["pi"]  # (648, 548)
    print(pi.shape)
    #计算最大值PImax,不指定axis，从每个维度找最大值
    pimax=np.max(pi)
    pimin=np.min(pi)
    #计算颜色分隔值
    pidef=pimin+(pimax-pimin)/2
    print("max:",pimax,"min",pimin,"def",pidef)
    ##彩色编码
    img=np.zeros((648,548,3),int)
    for x in range(0,648):
        for y in range(0,548):
            if(pi[x][y]<=pidef):
                img[x,y,:]=[0,int(255*(pi[x][y]-pimin)/(pimax-pimin)),255]
            else:
                img[x,y,:]=[255,int(255*(pimax-pi[x][y])/(pimax-pimin)),0]
    display(img)
    mp.imsave("D://output/0.png",img)          
    
    
  