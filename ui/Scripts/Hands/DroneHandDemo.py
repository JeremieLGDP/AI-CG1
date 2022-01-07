import torch
import joblib
import torch.nn as nn
import numpy as np
import cv2
from cvzone.HandTrackingModule import HandDetector
import torch.nn.functional as F
import time
import CustomModelCNN
 
# load label binarizer
lb = joblib.load('output/lb.pkl')
model = CustomModelCNN.CustomCNN()
model.load_state_dict(torch.load('output/best.pth'))
print(model)
print('Model loaded')

detector = HandDetector(detectionCon=0.8, maxHands=1)


def hand_area(img, x, y, w, h):
    if w<224:
        w=224
    if h<224:
        h=224
    w1=max(int(x-w/2),0)
    w2=max(int(x+w/2),0)
    h1=max(int(y-h/2),0)
    h2=max(int(y+h/2),0)
    hand = img[w1:w1+224, h1:h1+224]
    hand = cv2.resize(hand, (224,224))
    return hand


cap = cv2.VideoCapture(0)
if (cap.isOpened() == False):
    print('Error while trying to open camera. Plese check again...')
# get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# define codec and create VideoWriter object
out = cv2.VideoWriter('output/asl2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width,frame_height))


while(cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    hands, frame = detector.findHands(frame)

    if hands:
        hand1 = hands[0]
        centerPoint1 = hand1["center"] #Center of the hand cx,cy
        bbox1 = hand1["bbox"] # BBox info: x, y, w, h
        handType1 = hand1["type"] # left or right hand
    else:
        bbox1 = [112,112,224,224]

    # get the hand area on the video capture screen
    #cv2.rectangle(frame, (100, 100), (324, 324), (20, 34, 255), 2)
    hand = hand_area(frame,bbox1[0],bbox1[1],bbox1[2],bbox1[3])
    image = hand
    
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image = torch.tensor(image, dtype=torch.float)
    image = image.unsqueeze(0)
    
    outputs = model(image)
    _, preds = torch.max(outputs.data, 1)
    
    cv2.putText(frame, lb.classes_[preds], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    #plt.imshow(frame, cmap = 'gray', interpolation = 'bicubic')
    #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    #plt.show()
    cv2.imshow('image', frame)
    #exit()
    out.write(frame)
    # press `q` to exit
    if cv2.waitKey(27) & 0xFF == ord('q'):
        break
# release VideoCapture()
cap.release()
# close all frames and video windows
cv2.destroyAllWindows()