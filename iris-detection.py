import cv2 as cv
import numpy as np
import mediapipe as mp


capture = cv.VideoCapture(0)

while True:
    _, frame = capture.read()
    cv.imshow("Iris detection", cv.flip(frame, 1))
    key = cv.waitKey(1)
    if key == ord('q'):
        break
capture.release()
cv.destroyAllWindows()