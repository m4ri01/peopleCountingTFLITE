import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import paho.mqtt.publish as publish
import datetime
import json
intervalSend = 600
intervalAcquire = 60
peopleDetectTotal = []
peopleDetect1M = {}

webcam = cv2.VideoCapture(2)
imW = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
imH = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
MODEL_NAME = "saved_model/"
GRAPH_NAME = "people.tflite"
LABELMAP_NAME = "names.txt"
min_conf_threshold = float(0.5)

pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter

CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)
PATH_TO_LABELS = os.path.join(CWD_PATH,LABELMAP_NAME)
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5
outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname): # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

frame_rate_calc = 1
freq = cv2.getTickFrequency()

timeStart1 = datetime.datetime.now()
timeStart10 = datetime.datetime.now()
countMinute = 1
while True:
    t1 = cv2.getTickCount()
    _,frame1 = webcam.read()
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std
    # print(input_details[0]['index'])
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects
    jumlahOrang = 0
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0) and classes[i]==0):
            jumlahOrang+=1
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
            object_name = labels[int(classes[i])]
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
    cv2.rectangle(frame, (10,420), (320,460), (255,255,255), -1)
    cv2.putText(frame,"Jumlah Orang: {}".format(jumlahOrang), (20,450), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 3)
    peopleDetectTotal.append(jumlahOrang)
    timeNow1 = datetime.datetime.now()
    timeNow10 = datetime.datetime.now()
    timeDelta1 = timeNow1 - timeStart1
    timeDelta10 = timeNow10 - timeStart10
    if timeDelta1.seconds >= intervalAcquire:
        timeStart1 = datetime.datetime.now()
        listCalculate = np.array(peopleDetectTotal)
        dictVal = {"max":int(np.max(listCalculate)),"min":int(np.min(listCalculate)),"mean":float(np.mean(listCalculate))}
        peopleDetect1M[str(countMinute)] = dictVal
        print(dictVal)
        peopleDetectTotal = []
        countMinute+=1
    if timeDelta10.seconds >= intervalSend:
        timeStart10 = datetime.datetime.now()
        listCalculate = np.array(peopleDetectTotal)
        dictVal = {"max":int(np.max(listCalculate)),"min":int(np.min(listCalculate)),"mean":float(np.mean(listCalculate))}
        peopleDetect1M[str(countMinute)] = dictVal
        countMinute = 0
        publish.single("/people/counting",json.dumps(peopleDetect1M),hostname="localhost")
        peopleDetectTotal = []
        peopleDetect1M = {}

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    # Press 'q' to quit
    if cv2.waitKey(27) == ord('q'):
        break

cv2.destroyAllWindows()
webcam.release()