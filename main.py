import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture('video_1.mp4')
ret, frame = cap.read()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

pose_model = YOLO('yolov8n-pose.pt')

while ret:

    poses = pose_model.track(frame)[0].plot()

    poses = cv2.resize(poses, (640,480))

    out.write(poses)

    cv2.imshow('video', poses)
    if cv2.waitKey(25) == 27:
        break
    ret, frame = cap.read()

out.release()
cap.release()
cv2.destroyAllWindows()
