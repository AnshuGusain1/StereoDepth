import cv2

cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)

while cap0.isOpened() and cap1.isOpened():
    success0, img0 = cap0.read()
    success1, img1 = cap1.read()

    if not success0 or not success1:
        print("Failed to capture images from cameras.")
        break

    k = cv2.waitKey(5)

    if k == 27:  # Exit on 'ESC' key
        break
    elif k == ord('s'):  # Save images on 's' key
        cv2.imwrite('images/images_left/img_left.png', img0)
        cv2.imwrite('images/images_right/img_right.png', img1)
        print("Images saved!")

    # Draw horizontal line across the middle of each frame
    cv2.line(img0, (0, 540), (1920, 540), (255,0, 0), 10)
    cv2.line(img1, (0, 540), (1920, 540), (255,0, 0), 10)

    #img0 = cv2.resize(img0, (640, 480))
    #img1 = cv2.resize(img1, (640, 480))
    cv2.imshow('Left Camera', img0)
    cv2.imshow('Right Camera', img1)