import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

imgL = cv.imread('StereoDepth/images/images_right/img2.png', cv.IMREAD_GRAYSCALE)
imgR = cv.imread('StereoDepth/images/images_left/img2.png', cv.IMREAD_GRAYSCALE)


imgL = cv.resize(imgL, (640, 480))
imgR = cv.resize(imgR, (640, 480))

stereo = cv.StereoBM.create(numDisparities=16, blockSize=5)
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()