import cv2
import os
import pickle
import numpy as np
import mediapipe as mp

model_dict=pickle.load(open('./model/model.p','rb'))
model = model_dict['model']

mp_drawing=mp.solutions.drawing_utils
mp_hands=mp.solutions.hands

hands=mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=1)

cap = cv2.VideoCapture(0)

labels_dict={0: 'Pause', 1: 'Play', 2: 'Next', 3: 'Previous'}

while True:
    success, img = cap.read()
    frame = cv2.resize(img, (640, 480))
    temp = []
    tempX = []
    tempY = []
    if success:
        results = hands.process(frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:        
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )
            
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x,y  = hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y
                    temp.append(x)
                    temp.append(y)
                    tempX.append(x)
                    tempY.append(y)
                    
            x1 = int(min(tempX)*640)
            y1 = int(min(tempY)*480)
            
            x2 = int(max(tempX)*640)
            y2 = int(max(tempY)*480)
                
            prediction = model.predict([np.asarray(temp)])    
            action = labels_dict[int(prediction[0])]
            
            #print(action)
            
            cv2.rectangle(frame, (x1,y1-10), (x2,y2),(0,0,0),4)
            cv2.putText(frame, action, (x1,y1), cv2.FONT_HERSHEY_SIMPLEX,1.3,(0,0,0),3,cv2.LINE_AA)
            
            
    cv2.imshow("Hands",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
            
        