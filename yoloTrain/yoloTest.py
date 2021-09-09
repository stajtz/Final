# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 22:34:52 2021

@author: pluto
"""

import cv2
import numpy as np




whT = 320
cap = cv2.VideoCapture(0)
confThreshold=0.6
nmsThreshold=0.4

classFile="classes.names"
classNames=[]

classes = []
with open(classFile, "r") as f:
    classes = f.read().splitlines()


modelConfiguration="yolov4-custom.cfg"
modelWeights="yolov4-custom_best.weights"

#modelConfiguration="yolov4-tiny-3lRandom.cfg"
#modelWeights="yolov4-tiny-3lRandom_best.weights"


net=cv2.dnn.readNet(modelConfiguration,modelWeights)




while True:
    key = cv2.waitKey(1) 
    success,img=cap.read()
    height, width, _ = img.shape
    if True:
        # görüntüden 4 boyutlu bir blob oluşturur. blob aynı yükseklik ve genişlikteki işlenmiş görüntü topluluğudur.
                    blob = cv2.dnn.blobFromImage(img, 1/255, (whT, whT), (0, 0, 0), swapRB=True, crop=False)
                    #bloobları networke input olarak verilmesini sağlar
                    net.setInput(blob)
    
                    #çıktıların tabakalarını ayarlar
                    output_layers_names = net.getUnconnectedOutLayersNames()
                    layerOutputs = net.forward(output_layers_names)
    
                    boxes = []
                    confidences = []
                    class_ids = []

                    for output in layerOutputs:
                        for detection in output:
                            #tespit edilen neslerin uyumluluk yüzdelerini ve isimlerini kaydetmek için array oluşturulur
                            scores = detection[5:]
                            class_id = np.argmax(scores)
                            confidence = scores[class_id]
                            #uyumluluk confThresholdan fazlaysa nesnenin çerçevesi boxes arrayine atılır, ismi ve uyumluluğu arraye atılır
                            if confidence > confThreshold:
                                center_x = int(detection[0] * width)
                                center_y = int(detection[1] * height)
                                w = int(detection[2] * width)
                                h = int(detection[3] * height)

                                yolox = int(center_x - w / 2)
                                yoloy = int(center_y - h / 2)

                                boxes.append([yolox, yoloy, w, h])
                                confidences.append((float(confidence)))
                                class_ids.append(class_id)

                    #maksimum olmayan noktalar belirlenir
                    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
                    #isim için yazı fontu ve belirlenen nesneler için rastgele renkler seçilir
                    font = cv2.FONT_HERSHEY_PLAIN
                    # uyumluluğu %50den fazla olan nesneler için oluşturulmuş index için isim, uyumluluk değişkenlere atılı, seçilen renk belirlenir ve dikdörtgeni çizilir
                    if len(indexes) > 0:
                        for i in indexes.flatten():
                            yolox, yoloy, w, h = boxes[i]
                            label = str(classes[class_ids[i]])
                            confidence = str(round(confidences[i], 2))
                            cv2.rectangle(img, (yolox, yoloy), (yolox + w, yoloy + h), (0,255,0), 2)
                            cv2.putText(img,  label + " " + confidence, (yolox, yoloy + 20), font, 2, (255, 255, 255), 2)

   
    
    cv2.imshow("Mask",img)

    
    
    if key == 27:
        break
cv2.destroyAllWindows()