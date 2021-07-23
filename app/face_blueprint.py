from flask import abort, request, redirect, url_for, jsonify
from flask import Blueprint, render_template
from app.func_face import createMobileNet, prediksiImg, detect_faces, checkDirectory, loadModel, IMG_DIM
from sklearn.linear_model import LogisticRegression
from datetime import datetime
import os,cv2
import time
import pickle
import numpy as np

from app.config import Config

face_blueprint = Blueprint('face_blueprint', __name__,
    template_folder='templates',
    static_folder='static', static_url_path='assets')

## model untuk deteksi wajah atau tidak ##
haar_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
ModelmobileNet = createMobileNet()
# model=loadModel('app/modelTR.pkl')

@face_blueprint.route('/predictById', methods=['POST'])
def predictById():
    nid = request.form['id']
    name = request.form['image_name']
    nmFile = 'app\\photos\\predictFace\\%s\\%s'%(nid, name)
    model_ = loadModel('app/photos/model/'+str(nid)+'.pkl')
    
    t,r=prediksiImg(nmFile,nid,model_, ModelmobileNet, haar_face_cascade)
    elapsed = time.time() - t
    return "%s (Time Elapsed = %g)"%(r,elapsed)

@face_blueprint.route('/trainById', methods=['POST'])
def trainById():
    ## ambil data face terdahulu ##
    ftr_np = None
    nrp_np = None
    npzFILE = "app/dummy.npz"
    if os.path.exists(npzFILE):
        dataLoaded = np.load(npzFILE)
        ftr_np=dataLoaded['ftr_np']
        nrp_np =dataLoaded['nrp_np']

    t = time.time()

    nid = request.form['id']
    nrp_list_ = []
    ftr_list_ = []
    nrp=nid
    path = "app\\photos\\uploadFace\\" + nrp

    ## ambil semua foto dari folder nrp tersebut ##
    for imgFile in os.listdir(path):
        nmFile = path + "\\" + imgFile
        img = cv2.imread(nmFile)
        img = cv2.resize(img, IMG_DIM)

        ## coba cek apakah bener wajah di foto tersebut ##
        img = detect_faces(haar_face_cascade,img)
        if img is None:
            print("face not detected %s"%nmFile)
            continue
        img = cv2.resize(img, IMG_DIM)
        img=img/255
        img=img.reshape(1,img.shape[0],img.shape[1],img.shape[2])
        features = ModelmobileNet.predict(img, verbose=0)

        ftr_list_.append(features)
        nrp_list_.append(nrp)
    
    nrp_np_ = np.array(nrp_list_)
    ftr_np_ = np.array(ftr_list_)
    ftr_np_ = ftr_np_.reshape(ftr_np_.shape[0],ftr_np_.shape[2])

    ftr_np = np.concatenate((ftr_np,ftr_np_),axis=0)
    nrp_np = np.concatenate((nrp_np,nrp_np_),axis=0)

    # ## memindahkan foto-foto orang terkait ke folder trainedFace ##
    # now = datetime.now()
    # pathDEST = "app\\photos\\trainedFace\\%s_%s"%(nrp,now.strftime("%Y_%m_%d_%H_%M_%S"))
    # checkDirectory(pathDEST)
    # os.system('move %s %s'%(path,pathDEST))        
    
    if len(np.unique(nrp_np)) ==1:
        return "Labels only one class, cant create model"
    else:
        model_ = LogisticRegression(solver='lbfgs',n_jobs=-1, multi_class='auto',tol=0.8)
        model_.fit(ftr_np,nrp_np)
        # model = modelLR
        with open('app/photos/model/'+str(nrp)+'.pkl', 'wb') as f:
            pickle.dump(model_, f)
        elapsed = time.time() - t
        return "Save Model succeded Time Elapsed = %g"%elapsed



    