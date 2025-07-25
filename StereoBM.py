import numpy as np
import cv2 as cv

# Open left and right cameras (usually 0 and 1)
capL = cv.VideoCapture(0)
capR = cv.VideoCapture(1)

# Set resolution (optional)
capL.set(cv.CAP_PROP_FRAME_WIDTH, 640)
capL.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
capR.set(cv.CAP_PROP_FRAME_WIDTH, 640)
capR.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

# StereoSGBM settings
stereo = cv.StereoSGBM.create(
    minDisparity=0,
    numDisparities=32,
    blockSize=5,
    P1=8*3*5**2,
    P2=32*3*5**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()

    if not retL or not retR:
        print("Error capturing frames.")
        break

    grayL = cv.cvtColor(frameL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(frameR, cv.COLOR_BGR2GRAY)

    disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

    # Normalize for display
    disp_display = cv.normalize(disparity, None, 0, 255, cv.NORM_MINMAX)
    disp_display = np.uint8(disp_display)

    cv.imshow('Left Camera', frameL)
    cv.imshow('Right Camera', frameR)
    cv.imshow('Disparity', disp_display)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

capL.release()
capR.release()
cv.destroyAllWindows()