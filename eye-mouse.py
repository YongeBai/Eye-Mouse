import cv2 as cv
import mouse
import numpy as np
import mediapipe as mp
import torch
import sys


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

capture = cv.VideoCapture(s)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True,
                                  # 0.0-1.0 min confidence from face detection to be
                                  # considered successful
                                  min_tracking_confidence=0.5,
                                  # 0.0-1.0 higher values more accurate more latency (hyper param)
                                  min_detection_confidence=0.5)
left_iris = [469, 470, 471, 472]
n = 0
prev_loc = []
mouse.move(800, 450)

while True:
    ret, frame = capture.read()
    # frame = cv.flip(frame, 1)
    frame = cv.resize(frame, dsize=(1600, 900))
    height, width, _ = frame.shape

    if ret is False:
        break

    frame.flags.writeable = False
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        face_mesh_points = np.array([np.multiply([point.x, point.y], [width, height]).astype(int)
                                     for point in results.multi_face_landmarks[0].landmark])

        (x, y), _ = cv.minEnclosingCircle(face_mesh_points[left_iris])

        prev_loc.append((x, y))
        try:
            diff = tuple(map(lambda i, j: i - j, prev_loc[n], prev_loc[n+1]))
            mouse.move(diff[0] * 50, diff[1] * -50, absolute=False)

            print(f"mouse position{mouse.get_position()}")
            print(f"x_diff {diff[0]}")
            print(f"y_diff {diff[1]}")
        except:
            continue
        n += 1

        # center1 = (int(x1), int(y1))
        # cv.circle(frame, center1, int(radius1), (255, 255, 0), 1, cv.LINE_AA)


        # mouse.move(x_diff*20, y_diff*20, absolute=False)
        # print(f"x_diff: {x_diff}")
        # print(f"y_diff: {y_diff}")
        # print(mouse.get_position())


        # cv.polylines(frame, [face_mesh_points[left_eye]], True, (0, 255, 0), 1, cv.LINE_AA)

        # x1, x2 = face_mesh_points[471][0], face_mesh_points[469][0]
        # y1, y2 = face_mesh_points[470][1], face_mesh_points[472][1]
        # print(f"x-cords: {x1},{x2}")
        # print(f"y-cords: {y1},{y2}")
        # (x, y, w, h) = cv.boundingRect(face_mesh_points[left_iris])
        # cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # print(f"x-cords: {x},{x+w}")
        # print(f"y-cords: {y},{y+h}")
        # print(f"center cords: {(2*x+w)//2},{(2*y+w)//2}")
        #
        # frameCropped = frame[x:x+w, y:y+h]
        # grayframe = cv.cvtColor(frameCropped, cv.COLOR_RGB2GRAY)
        # cropResize = cv.resize(grayframe, dsize=(1280, 720))


        # (fx, fy), _ = cv.minEnclosingCircle(cropResize)
        # print(f"center cords: {fx},{fy}")

        # (x, y, w, h) = cv.boundingRect(cropResize)
        # fx = x + w/2
        # fy = x + h/2

        # (fx, fy), _= cv.minEnclosingCi1rcle(face_mesh_points[left_iris])
        #
        # mouse.move(fx, fy, absolute=True)
        # print(mouse.get_position())


        # mouse movements
        # print(mouse.get_position())
        # print(x, y)

        # print(mouse.get_position())
        # pyautogui.moveTo(x, y)
        # pyautogui.moveRel(0, 1)
        #



    cv.imshow("Iris detection", frame)
    #
    # cv.imshow("EYE resize", cropResize)
    # cv.imshow("gray", grayframe)

    if cv.waitKey(1) & 0xff == ord('q'):
        break
capture.release()
cv.destroyAllWindows()
