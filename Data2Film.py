import numpy as np
from PIL import Image

import cv2
import os


"""
async def read_image(X, val):
    image = Image.new("RGB", (256, 256))
    for i in range(256):
        for j in range(256):
            R = X[val][0][i][j]
            G = X[val][1][i][j]
            B = X[val][2][i][j]
            image.putpixel((i, j), (R, G, B))
    return image, val
"""


if __name__ == '__main__':
    with open("CEUS.bin", 'rb') as f:
        X = f.read()
        X = np.frombuffer(X, dtype=np.uint8)
        X = np.reshape(X, (200, 3, 256, 256))
        # print(X[0])
        # arr = np.array(X[0])

        image = Image.new("RGB", (256, 256))
        video_dir = 'D:/Study/超声造影/代码/DrawPicture/'
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        fps = 20
        video_dir = os.path.join(video_dir, "Video2.avi")
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

        video_writer = cv2.VideoWriter(video_dir, fourcc, fps, (256, 256))


        for x in range(200):
            for i in range(256):
                for j in range(256):
                    R = X[x][0][i][j]
                    G = X[x][1][i][j]
                    B = X[x][2][i][j]
                    image.putpixel((i, j), (R, G, B))
            #image.show()

            #先保存再读取会慢一点
            #f = image.save("C:/Test/Temp.bmp")
            #video_writer.write(cv2.imread("C:/Test/Temp.bmp"))

            f = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            video_writer.write(f)
            print("frame "+str(x)+" has been written")
        video_writer.release()
