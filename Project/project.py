# .pb file stands for protobuf. In TensorFlow, the protbuf file contains the graph definition as well as the weights
# of the model.
# Due to this file,we need not to train our dataset
# coco.names is the name of our classes that we can detect
# .pbtxt contains the configuration(light-weight). one of the best methods which has a good speed
import cv2
thres = 0.5
#url = "192.168.162.100:8080/video"#Threshold to detect object
cap = cv2.VideoCapture(0) #Opens camera for video capturing
cap.set(3, 1280)
cap.set(4, 720)#Dimensions
#cap.set(10, 70)
#img=cv2.imread("lena15.jpg")

classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:  # rt= read a file as text
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)#Neural
# Create detection model from network represented in one of the supported formats.
#Detection Model creates net from file with trained weights and config, sets preprocessing input,
#runs forward pass and return result detections.
#It return ids of classes from where we accesses the class name
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
#Set flag swapRB for frame.
#swapRB	Flag which indicates that swap first and last channels.

while True:
    success, img = cap.read() # success is a boolean regarding whether or not there was a return at all
    #and the img is each frame that is returned. If there is no frame, you wont get an error, you will get None or NoneType.
    classIds, confs, bbox = net.detect(img, confThreshold=thres) 
    #Given the input frame, create input binary large objects, run net and return result detections.
    #confs is basically in terms of a score that is the number of standard deviations away from the mean
    print(classIds, bbox)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
            cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    cv2.imshow("Output", img)
    cv2.waitKey(1)
