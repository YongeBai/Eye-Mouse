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
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=False,
                                  # 0.0-1.0 min confidence from face detection to be
                                  # considered successful
                                  min_tracking_confidence=0.96,
                                  # 0.0-1.0 higher values more accurate more latency
                                  min_detection_confidence=0.96)
x_scale = 30
y_scale = 45


points = [198,236,3,195,248,456,420,360,344,438,309,250,462,370,94,141,242,20,79,218,115,131,134,51,5,281,363,440,456,459,458,461,354,19,125,241,238,239,237,220,45,4,275,274,1,44,237,45,4]

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

        x_diff = prev_loc[0] - x
        y_diff = y - prev_loc[1]
        mouse.move(x_diff * x_scale, y_diff * y_scale, absolute=False, duration=0.1)

        print(f"mouse position{mouse.get_position()}")
        print(f"x_diff {x_diff}")
        print(f"y_diff {y_diff}")

        prev_loc = (x, y)
        n += 1

    cv.imshow("Iris detection", cv.flip(frame, 1))

    if cv.waitKey(1) & 0xff == ord('q'):
        break

capture.release()
cv.destroyAllWindows()
