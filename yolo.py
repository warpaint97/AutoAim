# YOLO object detection
import cv2 as cv
import numpy as np
import time

class YOLO:
    def __init__(self, use_gpu=False):
        # Give the configuration and weight files for the model and load the network.
        self.net = cv.dnn.readNetFromDarknet('model/yolov3.cfg', 'model/yolov3.weights')
        if not use_gpu:
            self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
            #net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
        else: 
            self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

        # determine the output layer
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        # Load names of classes and get random colors
        self.classes = open('model/coco.names').read().strip().split('\n')
        np.random.seed(0)
        self.colors = np.random.randint(0, 255, size=(len(self.classes), 3), dtype='uint8')

    def findPersonTarget(self, img, confidenceLevel=0.5, display_img=False, display_bboxes=True, keep_open=False):
        display_bboxes = False if not display_img else display_bboxes
        # construct a blob from the image
        blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
        # forward propagation
        self.net.setInput(blob)
        outputs = self.net.forward(self.ln)

        boxes = []
        confidences = []
        classIDs = []
        h, w = img.shape[:2]

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > confidenceLevel:
                    box = detection[:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    box = [x, y, int(width), int(height)]
                    boxes.append(box)
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        indices = cv.dnn.NMSBoxes(boxes, confidences, confidenceLevel, 0.4)

        preds_dict = {}
        if len(indices) != 0:
            for i in indices.flatten():
                key = self.classes[classIDs[i]]
                value = {
                    'confidence': confidences[i],
                    'position': (boxes[i][0], boxes[i][1]),
                    'size': (boxes[i][2], boxes[i][3])
                }
                preds_dict.setdefault(key, []).append(value)


        key = 'person'
        target = None
        if key in preds_dict.keys():
            target = list(preds_dict[key][0]['position'])
            target[0] += int(preds_dict[key][0]['size'][0]/2.0)
            target[1] += int(preds_dict[key][0]['size'][1]*0.25)

            # draw image
            if display_bboxes:
                color = [int(c) for c in self.colors[self.classes.index(key)]]
                for i, pred in enumerate(preds_dict[key]):
                    x, y = pred['position']
                    w, h = pred['size']
                    confidence = pred['confidence']
                    cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    text = f'[{i}]{key}: {confidence:.4f}'
                    cv.putText(img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    # draw target
                    cv.circle(img, tuple(target), 7, [0,0,255], 2)
        
        if display_img:
            cv.imshow('window', img)
            cv.waitKey(1) if not keep_open else cv.waitKey(0)

        # destroy deprecated objects
        del img
        del blob
        del outputs
        del boxes
        del confidences
        del classIDs
        del indices
        del preds_dict

        return target

# test
############################################################################################
if __name__ == '__main__':
    yolo = YOLO(use_gpu=False)
    target = yolo.findPersonTarget(cv.imread('images/test.jpg'), 0.5, display_img=True, display_bboxes=True, keep_open=True)
    print(f'Target: {target}')