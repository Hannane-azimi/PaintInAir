import cv2 as cv
import mediapipe as mp
import numpy as np
import math

window_name = "Finger Draw"
video = cv.VideoCapture(0)

if not video.isOpened():
    print("could not open camera")
    exit(1)

WIDTH = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))

mp_model = mp.solutions.hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils
model_connection = mp.solutions.hands.HAND_CONNECTIONS


canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

prev_x, prev_y = None, None


MIN_THICKNESS = 2
MAX_THICKNESS = 20
MIN_SPEED = 2
MAX_SPEED = 60

cv.namedWindow(window_name, cv.WINDOW_NORMAL)

while True:
    ret, frame = video.read()
    if not ret:
        break

    #frame = cv.flip(frame, 1) 

    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = mp_model.process(rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        index_tip = hand_landmarks.landmark[8]
        index_x = int(index_tip.x * WIDTH)
        index_y = int(index_tip.y * HEIGHT)

        if prev_x is not None and prev_y is not None:
            speed = math.hypot(index_x - prev_x, index_y - prev_y)

            
            speed_clamped = max(MIN_SPEED, min(MAX_SPEED, speed))
            ratio = (speed_clamped - MIN_SPEED) / (MAX_SPEED - MIN_SPEED)
            thickness = int(MAX_THICKNESS - ratio * (MAX_THICKNESS - MIN_THICKNESS))

            cv.line(canvas, (prev_x, prev_y), (index_x, index_y), (30, 30, 230), thickness)

        prev_x, prev_y = index_x, index_y

        cv.circle(frame, (index_x, index_y), 8, (30, 230, 30), -1)
    else:
        
        prev_x, prev_y = None, None

    
    gray = cv.cvtColor(canvas, cv.COLOR_BGR2GRAY)
    _, mask = cv.threshold(gray, 10, 255, cv.THRESH_BINARY)
    mask_inv = cv.bitwise_not(mask)
    frame_bg = cv.bitwise_and(frame, frame, mask=mask_inv)
    combined = cv.add(frame_bg, canvas)

    cv.imshow(window_name, combined)

    key = cv.waitKey(1) & 0xFF
    if key == 27 or not cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE):
        break
    if key == ord('c'):
        canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

mp_model.close()
video.release()
cv.destroyAllWindows()