import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

fingers = [0 for i in range(5)]

while True:
    _, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                if id == 4: # thumb tip
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
                if id == 8: # index finger tip
                    cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
        
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            # to check tip points of fingers
            for i in range(2,6):
                if handLms.landmark[i*4].y < handLms.landmark[(i*4)-2].y:
                    fingers[i-1] = 1
                else:
                    fingers[i-1] = 0
            
            # to check tip points of thumbs
            if handLms.landmark[4].x > handLms.landmark[5].x:
                fingers[0] = 1
            else:
                fingers[0] = 0

            # if hands stand reverse, change thumb decisions
            if handLms.landmark[8].x < handLms.landmark[20].x:
                fingers[0] = 1 - fingers[0]

            print(fingers)
            print(sum(fingers))

    img = cv2.flip(img, 1) # to see image like mirror
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()