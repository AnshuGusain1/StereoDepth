import cv2

cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)
num = 0

while cap0.isOpened():

    success, img0 = cap0.read()
    success1, img1 = cap1.read()

    img0 = cv2.resize(img0, (640, 480))
    img1 = cv2.resize(img1, (640, 480))

    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('images/images_left/img' + str(num) + '.png', img0)
        cv2.imwrite('images/images_right/img' + str(num) + '.png', img1)
        print("image saved!")
        num += 1

    cv2.imshow('Img',img0)

# Release and destroy all windows before termination
cap0.release()

cv2.destroyAllWindows()
