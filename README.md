# Eye-Mouse: Control your mouse cursor with your eye 

This project uses MediaPipe to detect and track the iris in a video stream and uses the iris position to move the mouse cursor on the screen. With Eye Mouse, you can control your cursor and interact with your computer simply by moving your eyes.

## Requirements
MediaPipe
OpenCV
Mouse

## Usage
To run the code, use the following command:

```
python eye_mouse.py
```
The code will automatically start processing frames from the video capture and moving the mouse cursor based on the detected iris position. To stop the code, press the 'q' key.

## Configuration
The code has several parameters that can be modified to change its behavior:

- `refine_landmarks`: Set this to `True` to enable landmark refinement. This can improve the accuracy of the detected landmarks.
- `min_tracking_confidence`: Set this to the minimum confidence level for tracking landmarks. Landmarks with a lower confidence will be discarded.
- `min_detection_confidence`: Set this to the minimum confidence level for detecting landmarks. Faces with a lower confidence will not be processed.
- `x_scale`: This is the scaling factor for the horizontal mouse movement. Increasing this value will make the mouse move more for the same iris movement.
- `y_scale`: This is the scaling factor for the vertical mouse movement. Increasing this value will make the mouse move more for the same iris movement.

## Limitations
The accuracy of the iris detection and tracking depends on the quality of the video frame and the lighting conditions. In low light or noisy environments, the detection and tracking may not work as well. The code also assumes that only one face is present in the frame, and it will only track the iris of the first detected face.
