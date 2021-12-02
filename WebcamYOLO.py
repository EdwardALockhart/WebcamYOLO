import cv2
from cv2.dnn import NMSBoxes
import numpy as np
import time
import multiprocessing

input_size = 320  # 320 for fast, 416, 512 for better but slower, 608 for best but slowest
# Download files to directory
# yolov3.weights https://pjreddie.com/darknet/yolo/
# yolov3.cfg https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
yolo_directory = "/YOLO/"

cv2.setUseOptimized(True)
cv2.setNumThreads(multiprocessing.cpu_count()-1)
config = yolo_directory + 'yolov3.cfg'
weights = yolo_directory + 'yolov3.weights'

coco_labels = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
               'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor',
               'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
               'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
               'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
               'toothbrush']
colors = np.random.randint(0, 255, size = (len(coco_labels), 3), dtype = "uint8")

net = cv2.dnn.readNetFromDarknet(config, weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
layer_names = net.getLayerNames()
layer_names  = [layer_names [i[0] - 1] for i in net.getUnconnectedOutLayers()]

cv2.startWindowThread()
capture = cv2.VideoCapture(0)
time.sleep(1)
count = 0
while count < 600:
    success, image = capture.read()
    if success == True:
        
        # Process and detect
        (image_height, image_width) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, scalefactor = (1/255.0), size = (input_size, input_size), swapRB = True, crop = False)
        net.setInput(blob)
        outputs = net.forward(layer_names)
        geometries, pos_probs, classes, class_ids = [], [], [], []
        for detections in outputs:
            for detection in detections:
                scores = detection[5:]
                label_id = np.argmax(scores)
                prob = scores[label_id]
                if prob > 0.5:
                    box = detection[0:4]*np.array([image_width, image_height, image_width, image_height])
                    (center_x, center_y, width, height) = box.astype("int")
                    x = center_x - (width/2)
                    y = center_y - (height/2)
                    x, y, width, height = [int(i) for i in [x, y, width, height]]
                    geometries.append([x, y, width, height])
                    pos_probs.append(prob)
                    classes.append(coco_labels[label_id])
                    class_ids.append(label_id)
        
        # Get the indices of the boxes to be kept
        indices = NMSBoxes(geometries, [float(i) for i in pos_probs], 0.7, 0.2)
        # Draw the final bounding boxes
        for i in indices:
            i = int(i)
            x, y, width, height = geometries[int(i)]
            color = [int(c) for c in colors[class_ids[i]]]
            cv2.rectangle(image, (x, y), (x + width, y + height), color, 2)
            text = "{}: {:.2f}".format(coco_labels[class_ids[i]], pos_probs[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imshow("image", image)
        cv2.waitKey(1)
        count += 1
capture.release()
cv2.destroyAllWindows()
