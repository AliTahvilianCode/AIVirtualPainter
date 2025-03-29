import cv2
import numpy as np
import time
import mediapipe as mp


xp, yp = 0, 0
x1, y1 = 0, 0


cap = cv2.VideoCapture(0)
ret, originalImage = cap.read()
m ,n ,t = originalImage.shape
imgCanvas = np.zeros((m, n, t), np.uint8)


mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils
pTime = 0
cTime = 0
tiy = 0
iy = 0


while True:
    ret, originalImage = cap.read()
    img = cv2.flip(originalImage, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h, w, c = img.shape
                # cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy) 

                if id == 5:
                    iy = 0
                    ix, iy = int(lm.x * w), int(lm.y * h)
                if id == 8:

                    xp, yp = x1, y1
                    tiy = 0
                    tix, tiy = int(lm.x * w), int(lm.y * h)
                    x1, y1 = tix, tiy

                    iy = int(iy)
                    tiy = int(tiy)

                    if tiy < iy:
                        index = 2
                    elif tiy > iy:
                        index = 1
                    else:
                        index = 0
                if id == 9:
                    my = 0
                    mx, my = int(lm.x * w), int(lm.y * h)
                if id == 12:
                    tmy = 0
                    tmx, tmy = int(lm.x * w), int(lm.y * h)
                    my = int(my)
                    tmy = int(tmy)
                    if tmy < my:
                        middle = 2
                    elif tmy > my:
                        middle = 1
                    else:
                        middle = 0

                if id == 13:
                    ry = 0
                    rx, ry = int(lm.x * w), int(lm.y * h)
                if id == 16:
                    rmy = 0
                    rmx, rmy = int(lm.x * w), int(lm.y * h)
                    ry = int(ry)
                    rmy = int(rmy)
                    if rmy < ry:
                        ring = 2
                    elif rmy > ry:
                        ring = 1
                    else:
                        ring = 0



                    if index == 2:
                        if middle == 1:
                            #print('draw')
                            cv2.putText(img, 'draw', (100, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)
                            cv2.circle(img, (x1, y1), 15, (255, 0, 0), cv2.FILLED)
                            cv2.line(imgCanvas, (xp, yp), (x1, y1), (0, 0, 255), 3)
                    if middle == 2 and ring == 1:
                        #print('sellect')
                        cv2.putText(img, '', (100, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)
                        cv2.circle(img, (x1, y1), 25, (0, 0, 0), cv2.FILLED)
                        cv2.line(imgCanvas, (xp, yp), (x1, y1), (0, 0, 0), 50)
                 #   else:
                        #print('nothing')
                        #cv2.putText(img, 'nothing', (100, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                #(255, 0, 255), 3)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)


    if cv2.waitKey(5) & 0xff == ord('q'):
        break
    cv2.imshow("img", img)
    cv2.waitKey(1)