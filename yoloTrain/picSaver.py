# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 05:19:51 2021

@author: pluto
"""
import cv2

cap=cv2.VideoCapture(0)
picCount=0


folderName="Resimler"
name="sprey"

while True:
    suc,frame=cap.read()
    key=cv2.waitKey(1)
    if suc:
        cv2.imshow("as",frame)
        if key==ord("s") or key==ord("+"):
            s=("{}/{}/{}{}.png").format(folderName,name,name,picCount)
            print(picCount)
            cv2.imwrite(s, frame)
            picCount=picCount+1
    if key==ord("q"):break
cv2.destroyAllWindows()
cap.release()