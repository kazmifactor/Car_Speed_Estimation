from ultralytics import YOLO
import cv2
from time import time
import numpy as np

# getting Video
cap = cv2.VideoCapture("input_video/input_video1.mp4")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
source_fps = int(cap.get(cv2.CAP_PROP_FPS))
print("frame_width: ", frame_width)
print("frame_height: ", frame_height)
print("Source FPS: ", source_fps)
if not cap.isOpened():
    print("Error Opening Video File.")

# Saving Video
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter("output_video/output_video.mp4", fourcc, source_fps, (frame_width, frame_height))


# Loading Yolo V8 Model
model = YOLO("models/yolov8n.pt")

# Road plyogons to be removed from tracking
road_polygon_points = [(0, 0), (0, int(0.28*frame_height)), (frame_width, int(0.28*frame_height)),
                       (frame_width, 0), (0, 0)]

ptime = 0
while True:
    rect, frame = cap.read()
    if not rect:
        break
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, [np.array(road_polygon_points, dtype=np.int32)], (255, 255, 255))
    original_region = cv2.bitwise_and(frame, mask)
    mask_inv = cv2.bitwise_not(mask)
    frame = cv2.bitwise_and(frame, mask_inv)

    # results = model.track(frame, conf=0.1, imgsz=(frame_width, frame_height),
    #                       persist=True, classes=[2, 7], tracker="bytetrack.yaml")
    # for result in results:
    #     for r in result.boxes.data.tolist():
    #         x1, y1, x2, y2, track_id, conf, class_id = r
    #         x1 = int(x1)
    #         y1 = int(y1)
    #         x2 = int(x2)
    #         y2 = int(y2)
    #         track_id = int(track_id)
    #         class_id = int(class_id)
    #
    #         cv2.rectangle(frame, (x1, y1), (x2, y2), (225, 0, 0), 2)
    #         cv2.putText(frame, ("car" if class_id == 2 else "truck" if class_id == 7 else None) + f" - {track_id}",
    #                     (x1, y1 - 10), 3, cv2.FONT_HERSHEY_PLAIN, (225, 0, 0), 2)

    frame = cv2.add(frame, original_region)
    ctime = time()
    fps = round((1 / (ctime - ptime)), 2)
    ptime = ctime
    cv2.putText(frame, f"FPS: {fps}", (30, 40), 4, cv2.FONT_HERSHEY_PLAIN, (0, 225, 0), 2)
    cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
    cv2.imshow("Output", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
