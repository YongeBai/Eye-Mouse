import cv2 as cv
import numpy as np
import dlib
import sys

s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]
capture = cv.VideoCapture(s)

detector = dlib.get_frontal_face_detector()

dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

left_eye = list(range(43, 48))
right_eye = list(range(37, 43))

while True:
    _, frame = capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        face_landmarks = dlib_facelandmark(gray, face)

        for i,j in zip(left_eye, right_eye):
            left_x = face_landmarks.part(i).x
            left_y = face_landmarks.part(i).y
            cv.circle(frame, (left_x, left_y), 1, (255, 0, 255), 2)

            right_x = face_landmarks.part(j).x
            right_y = face_landmarks.part(j).y
            cv.circle(frame, (right_x, right_y), 1, (255, 0, 255), 2)

    cv.imshow("Face detection", cv.flip(frame, 1))
    key = cv.waitKey(1)
    if key == ord('q'):
        break
capture.release()
cv.destroyAllWindows()
