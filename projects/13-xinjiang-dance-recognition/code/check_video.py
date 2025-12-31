import cv2
cap = cv2.VideoCapture(r'D:\py\PythonProject\xinjiangwu\demo\demo.mp4')
print('总帧数', int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
print('isOpened', cap.isOpened())
ret, frame = cap.read()
print('第一帧读取', ret, frame.shape if ret else 'None')
cap.release()
import numpy as np
a = np.load('npy/dance0001.npy')
print('x', a[:, 0::3].min(), a[:, 0::3].max())
print('y', a[:, 1::3].min(), a[:, 1::3].max())
print('z', a[:, 2::3].min(), a[:, 2::3].max())