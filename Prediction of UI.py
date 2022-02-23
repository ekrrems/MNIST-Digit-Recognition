import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import keyboard
import pickle



pickle_in = open("mnist_digit_model.p","rb")
model = pickle.load(pickle_in)



def pred(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(28,28))
    pred = model.predict(img[np.newaxis, :, :, np.newaxis])
    return str(np.argmax(pred.astype('int')))
    

# User Interface

drawing = False
ix,iy = -1,-1

def draw_line(event,x,y,flags,params):
    global drawing,ix,iy

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy =x,y
    elif event ==cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img,(ix,iy),10,(0,255,0),-1)
            cv2.circle(img,(x,y),10,(0,255,0),-1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(img,(x,y),10,(0,255,0),-1)
        
img = np.zeros((800,800,3),np.uint8)


cv2.namedWindow('Paint Brush')
cv2.setMouseCallback('Paint Brush',draw_line)


while 1:
    cv2.imshow('Paint Brush',img)
    
    if keyboard.is_pressed('p'):
        cv2.putText(img,pred(img),(225,225),
                    cv2.FONT_HERSHEY_DUPLEX,5,
                    color=(0,0,250),thickness=2)
        
    elif keyboard.is_pressed('c'):
        img = np.zeros((800,800,3),np.uint8)
        
    if cv2.waitKey(20) & 0XFF == ord('q'):
        break
        
cv2.destroyAllWindows()
        
