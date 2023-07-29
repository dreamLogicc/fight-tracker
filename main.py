import os
import cv2
import random
from ultralytics import YOLO
from tracker import Tracker
import numpy as np

cap = cv2.VideoCapture('video_1.mp4')
ret, frame = cap.read()

model = YOLO('yolov8n.pt')
tracker = Tracker()

colors = [(random.randint(0, 255), random.randint(0, 255),
          random.randint(0, 255)) for j in range(10)]

while ret:

    results = model(frame)

    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            class_id = int(class_id)
            detections.append([int(x1), int(y1), int(x2), int(y2), score])

        print(detections)
        tracker.update(frame=frame, detections=np.array(detections))

        for track in tracker.tracks:
            bbox = track.bbox
            track_id = track.track_id
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)

    cv2.imshow('video', frame)
    if cv2.waitKey(25) == 27:
        break
    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()
