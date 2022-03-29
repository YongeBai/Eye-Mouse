import cv2 as cv
import numpy as np
import mediapipe as mp
# win32api.SetCursorPos((x,y))

capture = cv.VideoCapture(0)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True,
                                  # 0.0-1.0 min confidence from face detection to be
                                  # considered successful
                                  min_tracking_confidence=0.5,
                                  # 0.0-1.0 higher values more accurate more latency (hyper param)
                                  min_detection_confidence=0.5)
left_iris = [474, 475, 476, 477]
right_iris = [469, 470, 471, 472]

while True:
    ret, frame = capture.read()
    height, width, _ = frame.shape
    if ret is False:
        break

    frame.flags.writeable = False
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        face_mesh_points = np.array([np.multiply([point.x, point.y], [width, height]).astype(int) for point in results.multi_face_landmarks[0].landmark])
        (left_cordx, left_cordy), radius_left = cv.minEnclosingCircle(face_mesh_points[left_iris])
        (right_cordx, right_cordy), radius_right = cv.minEnclosingCircle(face_mesh_points[right_iris])

        center_left = (int(left_cordx), int(left_cordy))
        radius_left = int(radius_left)

        center_right = (int(right_cordx), int(right_cordy))
        radius_right = int(radius_right)

        cv.circle(frame, center_left, radius_left, (0, 255, 0), 1, cv.LINE_AA)
        cv.circle(frame, center_right, radius_right, (0, 255, 0), 1, cv.LINE_AA)

    cv.imshow("Iris detection", cv.flip(frame, 1))
    key = cv.waitKey(1)
    if key == ord('q'):
        break
capture.release()
cv.destroyAllWindows()
