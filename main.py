import os
import cv2
import random
from ultralytics import YOLO
from tracker import Tracker
import numpy as np
import mediapipe as mp

cap = cv2.VideoCapture('video_1.mp4')
ret, frame = cap.read()

# cap_out = cv2.VideoWriter('out.mp4', cv2.VideoWriter_fourcc(
#     *'MP4V'), cap.get(cv2.CAP_PROP_FPS), (frame.shape[0], frame.shape[1]))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
cap_out = cv2.VideoWriter('output.avi',
                          fourcc,
                          30, (frame.shape[0], frame.shape[1]))

model = YOLO('yolov8n.pt')
tracker = Tracker()

pose_model = YOLO('yolov8n-pose.pt')

# Read the network into Memory
colors = [(random.randint(0, 255), random.randint(0, 255),
          random.randint(0, 255)) for j in range(3)]

while ret:

    results = model(frame)
    poses = pose_model(frame)[0]

    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            class_id = int(class_id)
            detections.append([int(x1), int(y1), int(x2), int(y2), score])

        tracker.update(frame=frame, detections=np.array(detections))

        for track in tracker.tracks:
            bbox = track.bbox
            track_id = track.track_id
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (int(x1), int(y1)), (int(
                x2), int(y2)), (colors[track_id % len(colors)]), 3)

    pose_frame = poses.plot()

    # cap_out.write(frame)
    cv2.imshow('video', pose_frame)
    if cv2.waitKey(25) == 27:
        break
    ret, frame = cap.read()

cap.release()
cap_out.release()
cv2.destroyAllWindows()
