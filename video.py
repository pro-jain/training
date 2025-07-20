import cv2
from ultralytics import YOLO

model = YOLO("C:\\Users\\GNC_LAB_5\\Documents\\Bodhini\\runs\\detect\\train33\\weights\\best.pt")  
cap=cv2.VideoCapture("rtsp://192.168.144.25:8554/main.264")

while cap.isOpened():
    ret,frame= cap.read()
    if not ret:
         break
    #image_path="c:\\Users\\GNC_LAB_5\\Documents\\Bodhini\\test\\smoke39.webp"
    #img = cv2.imread(image_path)

    results = model(frame,iou=0.6,conf=0.1)     
    annotated_img = results[0].plot()  
    #resized_output = cv2.resize(annotated_img, (640, 640))
    cv2.imshow('YOLO Detections', annotated_img)
    if cv2.waitKey(0):
        cv2.destroyAllWindows()
