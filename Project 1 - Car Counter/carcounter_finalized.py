from ultralytics import YOLO
import cv2 as cv
import cvzone
import numpy as np
import math
from sort import *

model = YOLO('../yolo_weights/yolov8n.pt')


classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ] 

mask = cv.imread('karachi.png')
mask = cv.resize(mask,(1920, 1080)) 


tracker = Sort(max_age=15,min_hits=2,iou_threshold=0.6)

capture = cv.VideoCapture('karachi.mp4')

crossedline = [800,600,1400,600]

car_detect_count = 0
id_list = []

frame_width = int(capture.get(3))
frame_height = int(capture.get(4))
   
size = (frame_width, frame_height)
out = cv.VideoWriter('filename.avi', cv.VideoWriter_fourcc(*'MJPG'),10, size)

while True:
    ret,frame = capture.read()
    print(frame.shape,mask.shape)
    cropped = cv.bitwise_and(frame,mask)
    result = model(cropped,stream = True) # stream = True , use generator objects
    dets=np.empty((0, 5)) # tracker.update is intialized as an empty array that changes after different frames
    
    for r in result: # r has tensors in it
        boxes = r.boxes
        for coord in boxes:
            xmin,ymin, xmax , ymax = coord.xyxy[0] # we can also use coord.xyxy
            xmin,ymin, xmax , ymax = int(xmin),int(ymin), int(xmax) , int(ymax)
            # cv.rectangle(frame,(xmin,ymin),(xmax,ymax),(220,100,50),2,cv.LINE_4)

            # cvzone provide customized bounding boxes.
            bbox = xmin,ymin, xmax-xmin , ymax-ymin # bbox is bounding box
            cvzone.cornerRect(frame,bbox=bbox,l = 10 , t = 2 , rt =1,colorR = (255,255,255),colorC = (0,0,0)) # (0,165,255)

            # to get the confidence level
            confidence = (math.ceil(coord.conf[0]*100))/100

            # for class name
            clsname = int(coord.cls[0])

            # cv.line(frame,(crossedline[0],crossedline[1]),(crossedline[2],crossedline[3]),color = (0,0,255),thickness=5)

            # putting the confidence level on the rectangle on the top left of the bbox
            cvzone.putTextRect(frame,f'{classNames[clsname]} {confidence}',(max(0,xmin),max(20,ymin)),scale = 1,thickness = 1,offset = 4,colorR = (0,0,0))
            arr = np.array([xmin,ymin,xmax,ymax,confidence])
            dets = np.vstack((dets,arr)) # appending the changes vertically 
                # car_detect_count += 1
                # cv.putText(frame,f'No. of cars:{car_detect_count}',(900,100),fontFace  = cv.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0,0,0),thickness=2)

    # setting up tracker
    trackres = tracker.update(dets)
    
    for t in trackres:
        xmin,ymin,xmax,ymax,id = t
        xmin,ymin,xmax,ymax,id = int(xmin),int(ymin),int(xmax),int(ymax),int(id)
        # print(t)
        w,h = xmax - xmin , ymax - ymin
        # cvzone.cornerRect(frame,(xmin,ymin,w,h),l = 10 , t = 1, rt = 1 ,colorR = (0,0,0),colorC = (255,255,255))
        # cvzone.putTextRect(frame,f'{id}',(max(0,xmin),max(20,ymin)),scale = 1,thickness = 1,offset = 10,colorR = (0,0,0))

        midpointx , midpointy = (xmin+xmax)//2,(ymin+ymax)//2
        # midpointx , midpointy = (xmin+w)//2,(ymin+h)//2

        # print(f'mid point is ({midpointx} , {midpointy})')
        # cv.circle(frame,(midpointx , midpointy),5  ,color = (0,0,0),thickness=-1)
        # cv.putText(frame,f'{midpointy}',(midpointx , midpointy),fontFace=cv.FONT_HERSHEY_PLAIN,fontScale=1,color = (0,0,0))

        # print(midpointy)
        if (crossedline[0] < midpointx < crossedline[2]) and crossedline[1] - 15 < midpointy < crossedline[3] + 15 :
            id_list.append(id)
            if id_list.count(id)==1:
                # cv.line(frame,(crossedline[0],crossedline[1]),(crossedline[2],crossedline[3]),color = (0,255,0),thickness=5)
                car_detect_count += 1
    cv.putText(frame,f'No. of Vehicles: {car_detect_count}',(900,100),fontFace  = cv.FONT_HERSHEY_PLAIN,fontScale=2.5,color=(0,0,0),thickness=2)
    out.write(frame)
        
    # cv.imshow('cropped',cropped)
    # cv.resizeWindow('cropped', 1920, 1080)
    cv.imshow('image',frame)
    cv.resizeWindow('image', 1920, 1080)
    cv.waitKey(1)
    # break
