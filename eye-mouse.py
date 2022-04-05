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
                                  min_tracking_confidence=0.96,
                                  # 0.0-1.0 higher values more accurate more latency
                                  min_detection_confidence=0.96)
x_scale = 25
y_scale = 35

left_iris = [469, 470, 471, 472]

n = 0
prev_loc = (800, 450)
mouse.move(prev_loc[0], prev_loc[1])

while True:
    ret, frame = capture.read()
    height, width, _ = frame.shape

    if ret is False:
        break

    frame.flags.writeable = False
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        face_mesh_points = np.array([np.multiply([point.x, point.y], [width, height]).astype(int)
                                     for point in results.multi_face_landmarks[0].landmark])

        (x, y), _ = cv.minEnclosingCircle(face_mesh_points[points])
        (x1, y1, w, h) = cv.boundingRect(face_mesh_points[left_iris])
        cv.rectangle(frame, (x1, y1), (x1+w, y1+h), (255, 0, 0), 2)

        x_diff = prev_loc[0] - x
        y_diff = y - prev_loc[1]
        mouse.move(x_diff * x_scale, y_diff * y_scale, absolute=False, duration=0.05)

        # print(f"mouse position{mouse.get_position()}")
        # print(f"x_diff {x_diff}")
        # print(f"y_diff {y_diff}")

        prev_loc = (x, y)
        n += 1

    cv.imshow("Iris detection", cv.flip(frame, 1))

    if cv.waitKey(1) & 0xff == ord('q'):
        break

capture.release()
cv.destroyAllWindows()
