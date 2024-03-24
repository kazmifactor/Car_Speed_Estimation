from ultralytics import YOLO
import cv2
from time import time
import numpy as np
from birds_eye_view import BirdsEyeView
import colorsys


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


def color(tracking_id):
    hue = (tracking_id * 137.5) % 360  # Change 137.5 to adjust the hue spread
    red, green, brown = colorsys.hsv_to_rgb(hue / 360, 1.0, 1.0)
    return int(red * 255), int(green * 255), int(brown * 255)


# Loading Yolo V8 Model
model = YOLO("models/yolov8n.pt")

# Road polygons to be removed from tracking
road_polygon_points = [(0, 0), (0, 789), (3300, int(789)),
                       (3300, 0), (0, 0)]

# Perspective Transform from Camera view to Birds Eye View
target_width = 50
target_height = 250
source = np.array([[1252, 789], [2289, 789], [5039, 2159], [-550, 2159]])
target = np.array([[0, 0], [target_width-1, 0], [target_width-1, target_height-1], [0, target_height-1]])
transformation = BirdsEyeView(source, target, frame_width, frame_height, target_width, target_height)


# Dictionary to store previous y-coordinate
prev_y_dict = {}

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
    frame = transformation.draw_road(frame)

    results = model.track(frame, conf=0.1, imgsz=(3840, 2176),
                          persist=True, classes=[2, 7], tracker="bytetrack.yaml", verbose=False)

    for result in results:
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, track_id, conf, class_id = r
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            track_id = int(track_id)
            class_id = int(class_id)
            color_id = color(track_id)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color_id, 5)
            cv2.putText(frame, ("car" if class_id == 2 else "truck" if class_id == 7 else None) + f" - {track_id}",
                        (x1, y1 - 10), 1, cv2.FONT_HERSHEY_COMPLEX, color_id, 3)

            # Transformation and drawing on canvas
            bottom_center_point_bbox = (int((x1 + x2) / 2), y2)
            cv2.circle(frame, bottom_center_point_bbox, 10, color_id, -1)
            bottom_center_point_array = np.array([bottom_center_point_bbox])
            transformed_points = transformation.transform_points(points=bottom_center_point_array)[0]
            frame = transformation.draw_car_point(transformed_points, frame, track_id, color_id)

            # Speed Calculation
            if track_id not in prev_y_dict:
                prev_y_dict[track_id] = [transformed_points[1]]
            else:
                prev_y_dict[track_id].append(transformed_points[1])
                speed = transformation.speed_calculation(prev_y_dict[track_id], source_fps)
                if len(prev_y_dict[track_id]) > (source_fps / 2):
                    prev_y_dict[track_id].pop(0)

                # Labeling Speed Calculation on the car and canvas
                label_width = int((x2 - x1) * 0.45)
                bbox_center = int(((x1 + x2) / 2))
                cv2.rectangle(frame, (bbox_center-label_width, y2+10), (bbox_center+label_width, y2 + 40), color_id, -1)
                if label_width > 115:
                    text_scale = 0.5
                else:
                    text_scale = 0.95
                cv2.putText(frame, f"{speed} Km/h.",
                            (int(bbox_center-label_width * text_scale), y2+33), 4, cv2.FONT_HERSHEY_PLAIN, (0, 0, 0), 2)
                transformation.label_speed_on_canvas(speed, frame, color_id)

    frame = cv2.add(frame, original_region)
    ctime = time()
    fps = round((1 / (ctime - ptime)), 2)
    ptime = ctime
    cv2.putText(frame, f"FPS: {fps}", (30, 40), 4, cv2.FONT_HERSHEY_PLAIN, (0, 225, 0), 2)
    cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
    cv2.imshow("Output", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
