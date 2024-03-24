import cv2
from time import time


cap = cv2.VideoCapture("input_video/input_video1.mp4")


if not cap.isOpened():
    print("Error Opening Video File.")

ptime = 0
while True:
    rect, frame = cap.read()
    if not rect:
        break

    ctime = time()
    fps = round((1 / (ctime - ptime)), 2)
    ptime = ctime
    cv2.putText(frame, f"FPS: {fps}", (30, 40), 4, cv2.FONT_HERSHEY_PLAIN, (0, 225, 0), 2)
    cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
    cv2.imshow("Output", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
