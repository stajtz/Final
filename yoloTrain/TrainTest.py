# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 06:05:31 2021

@author: pluto
"""

import os
import random

pathFolder="Resimler"
pathName="sprey"

trainPath="customData"

path=pathFolder+"/"+pathName
i=0
p=[]

for f in os.listdir(path):
    trainDir=trainPath+"/"+f+"\n"
    p.append(trainDir)

random.shuffle(p)
pTest=p[:int(len(p)*0.2)]
pTrain=p[int(len(p)*0.2):]
with open('train.txt', 'a') as train_txt:
    for e in pTrain:
        train_txt.write(e)
        
with open('test.txt', 'a') as test_txt:
    for e in pTest:
        test_txt.write(e)


