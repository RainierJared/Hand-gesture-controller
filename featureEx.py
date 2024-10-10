import os
import cv2

DATA_DIR='./data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    
gesture_number=4
dataset_size=900

cap = cv2.VideoCapture(0)
for i in range(gesture_number):
    if not os.path.exists(os.path.join(DATA_DIR, str(i))):
        os.makedirs(os.path.join(DATA_DIR,str(i)))
        
    print("Collecting data for class {}".format(i))
    
    
    while True:
        success,img = cap.read()
        img1 = cv2.resize(img, (640,480))
        cv2.putText(img1, "Press 'Q' to capture data", (100,50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,0),3, cv2.LINE_AA)
        cv2.imshow("Data Extract",img1)
        if cv2.waitKey(1) == ord('q'):
            break
        
    counter = 0
    while counter < dataset_size:
        _, frame = cap.read()
        frame1 = cv2.resize(frame, (640,480))
        cv2.imshow("Data Extract", frame1)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(i), '{}.jpg'.format(counter)),frame1)
        counter +=1
        
        
cap.release()
cv2.destroyAllWindows()
