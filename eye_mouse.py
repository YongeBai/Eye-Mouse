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

# create a face mesh solution for landmark detection and tracking
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True,
                                  # set minimum tracking confidence to 0.96
                                  min_tracking_confidence=0.96,
                                  # set minimum detection confidence to 0.96
                                  min_detection_confidence=0.96)

# set scaling factors for mouse movement
x_scale = 25
y_scale = 35

# define points to detect in face mesh
points = [198,236,3,195,248,456,420,360,344,438,309,250,462,370,94,141,242,20,79,218,115,131,134,51,5,281,363,440,456,459,458,461,354,19,125,241,238,239,237,220,45,4,275,274,1,44,237,45,4]
left_iris = [469, 470, 471, 472]

# initialize frame counter and previous mouse location
n = 0
prev_loc = (800, 450)

# move mouse to initial position
mouse.move(prev_loc[0], prev_loc[1])

# enter main loop to process video frames
while True:
    # read a frame from the video capture
    ret, frame = capture.read()

    # if there are no more frames, break from the loop
    if ret is False:
        break

    # make the frame read-only to prevent modifications
    frame.flags.writeable = False

    # convert frame to RGB color space
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # process the frame using the face mesh solution
    results = face_mesh.process(rgb_frame)

    # if iris landmarks are detected
    if results.multi_face_landmarks:
        # extract iris landmark points from the results
        face_mesh_points = np.array([np.multiply([point.x, point.y], [width, height]).astype(int)
                                     for point in results.multi_face_landmarks[0].landmark])

        # compute minimum enclosing circle around iris landmarks
        (x, y), _ = cv.minEnclosingCircle(face_mesh_points[points])

        # compute bounding box around left iris
        (x1, y1, w, h) = cv.boundingRect(face_mesh_points[left_iris])
        cv.rectangle(frame, (x1, y1), (x1+w, y1+h), (255, 0, 0), 2)

        x_diff = prev_loc[0] - x
        y_diff = y - prev_loc[1]
        
        # move mouse based on iris position difference and scaling factors
        mouse.move(x_diff * x_scale, y_diff * y_scale, absolute=False, duration=0.05)

        # print(f"mouse position{mouse.get_position()}")
        # print(f"x_diff {x_diff}")
        # print(f"y_diff {y_diff}")
        # update previous iris position
        prev_loc = (x, y)

        # increment frame counter
        n += 1
        
    # display the processed frame to the user
    cv.imshow("Iris detection", cv.flip(frame, 1))

    if cv.waitKey(1) & 0xff == ord('q'):
        break

capture.release()
cv.destroyAllWindows()
