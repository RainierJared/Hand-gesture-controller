import os
import cv2
import pickle
import mediapipe as mp
import matplotlib.pyplot as plt

DATA_DIR="./data"

data = []
labels = []

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=1)

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR,dir_)):
        temp = []
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = hands.process(rgbImg)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x,y  = hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y
                    temp.append(x)
                    temp.append(y)
                    
            data.append(temp)
            labels.append(dir_)

f = open('data.pickle','wb')
pickle.dump({'data': data, 'labels':labels}, f)
f.close()