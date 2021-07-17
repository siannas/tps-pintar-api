# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 15:52:39 2020

@author: Admin
"""
import pickle,sys
from keras.applications.mobilenet import MobileNet
DIM=224
IMG_DIM = (DIM, DIM)
input_shape = (DIM, DIM, 3)
import keras
from keras.models import Model
import os,cv2
import time
import operator
import ctypes
import sys


#hllDll = ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\bin")

def loadModel(nmModel):
    f = open(nmModel, 'rb')
    model = pickle.load(f)
    return model

def createMobileNet():
    mobileNet =  MobileNet(include_top=False, weights='imagenet', 
                                     input_shape=input_shape)

    output    = mobileNet.layers[-1].output
    output    = keras.layers.Flatten()(output)
    ModelmobileNet = Model(inputs=mobileNet.input, outputs=output)# base_model.get_layer('custom').output)

    ModelmobileNet.trainable = False
    for layer in ModelmobileNet.layers:
        layer.trainable = False
    return ModelmobileNet

def detect_faces(f_cascade, img, scaleFactor = 1.1,needGray=False):
    sub_face = None
    if needGray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=1);
    for (x, y, w, h) in faces:
        sub_face = gray[y+2:y+h-2, x+2:x+w-2]
    
    return sub_face

def prediksiImg(nmFile,nrp,model, ModelmobileNet, haar_face_cascade):
    t = time.time()
    img = cv2.imread(nmFile)
    if img is None:
        return t,"REJECTED, not valid file , cant be predict"

    ## SEPERTINYA TIDAK PERLU DIPREDIKSI WAJAH ATAU BUKAN ##
    img = cv2.resize(img, IMG_DIM)
    img = detect_faces(haar_face_cascade,img)
    if img is None:
        if nmFile.find('X__') > 0:
             nmFileNew = nmFile.replace('X__','NF__')
             os.system('move %s %s'%(nmFile,nmFileNew))
        return t,"REJECTED, not face no remove, cant be predict"

    img = cv2.resize(img, IMG_DIM)
    img=img/255
    img=img.reshape(1,img.shape[0],img.shape[1],img.shape[2])
    
    ftr_np = ModelmobileNet.predict(img, verbose=0)

    predicted_proba = model.predict_proba(ftr_np)
    res = {}
    prob = -1
    for i in range(len(model.classes_)):
        res[model.classes_[i]] = predicted_proba[0][i]
    res = sorted(res.items(), key=operator.itemgetter(1))
    res.reverse()

    rank = 0

    for key, val in res:
        rank += 1
        if key == nrp:
            prob = val
            break
    score = round(prob*100,2)
    nmFile = nmFile.replace('/','\\')
    if prob == -1:
        if nmFile.find('X__') > 0:
             nmFileNew = nmFile.replace('X__','NR__')
             os.system('move %s %s'%(nmFile,nmFileNew))
        return t,"REJECTED,ERR: Face data of " + nrp + " is currently not registered, no photos have been trained"


    if rank <= 5 and score > 60:
        result = "ACCEPTED,"
        if nmFile.find('X__')>0:
             nmFileNew = nmFile.replace('X__','A__')
        else:
             nmFileNew = nmFile.replace('R__','A__')
        cmd='move %s %s'%(nmFile,nmFileNew)
        os.system(cmd)


    else:
        result = "REJECTED,"
        if nmFile.find('X__') > 0:
             nmFileNew = nmFile.replace('X__','R__')
             os.system('move %s %s'%(nmFile,nmFileNew))

    return t,"%s %s mobileNet score %g  rank %g" %(result,nrp,score,rank)

def checkDirectory(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise