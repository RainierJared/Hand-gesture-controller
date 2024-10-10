import cv2
import os
import pickle
import time
import numpy as np
import mediapipe as mp
import streamlit as st
from pynput.keyboard import Key, Controller

labels_dict={0: 'Pause/Play', 1: 'Next', 2: 'Previous'}
model_dict=pickle.load(open('./model/model.p','rb'))
model = model_dict['model']

mp_drawing=mp.solutions.drawing_utils
mp_hands=mp.solutions.hands
hands=mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=1)

cap = cv2.VideoCapture(0)
keyboard = Controller()

oldPre='9'
prediction = '9'

#FOR THE WEBAPP
st.title("Webapp Gesture Recognition")

placeholder_frame = st.empty()

stop_button = st.button("Stop")


def spotifyController(prediction):

    localKeyboard = keyboard
    match prediction:
        case 0:
            localKeyboard.press(Key.space)
            localKeyboard.release(Key.space)
            
        case 1:
            localKeyboard.press(Key.ctrl_l)
            localKeyboard.press(Key.right)
            localKeyboard.release(Key.ctrl_l)
            localKeyboard.release(Key.right)
 
        case 2:
            localKeyboard.press(Key.ctrl_l)
            localKeyboard.press(Key.left)
            localKeyboard.release(Key.ctrl_l)
            localKeyboard.release(Key.left)

        case _:
            print("Unknown input")

def gestureRecog(frame):
    global oldPre, prediction
    temp = []
    tempX = []
    tempY = []
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
        
        cv2.rectangle(frame, (x1,y1-10), (x2,y2),(0,0,0),4)
        cv2.putText(frame, action, (x1,y1), cv2.FONT_HERSHEY_SIMPLEX,1.3,(0,0,0),3,cv2.LINE_AA)

    if oldPre != int(prediction[0]):     
        spotifyController(int(prediction[0]))
        oldPre = int(prediction[0])
        
while cap.isOpened() and not stop_button:
    success,img=cap.read()
    frame=cv2.resize(img,(640,480))
    
    if not success:
        st.write("Video Capture has ended")
        break
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    placeholder_frame.image(frame, channels="RGB")
    
    gestureRecog(placeholder_frame.image)
    
    cv2.imshow("Hands",frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or stop_button:
        break
    
cap.release()
cv2.destroyAllWindows()
            
        