# Code source by Ionuț Mironică
# This script evaluates all the algorithms on several test images

import pickle
import cv2
from FeatureExtraction import FeatureExtraction
import os
import numpy as np
from sklearn import preprocessing
import skimage.data
import skimage.transform
from keras.models import load_model
from scipy.cluster.vq import *

# feature extractors
featureExtraction = FeatureExtraction()
detector = cv2.xfeatures2d.SIFT_create()

# load mathematical models
modelRfGlcm = pickle.load(open('models/RF_GLCM.dat', 'rb'))
modelRfHsv = pickle.load(open('models/RF_HSV.dat', 'rb'))
modelRfLBP = pickle.load(open('models/RF_LBP.dat', 'rb'))
modelCNN = load_model('models/conv_model_4_8_256_0.1.dat')
modelSVMBowSift = pickle.load(open('models/BOW_SIFT_SVM.dat', 'rb'))
dictionarySVMBowSift = pickle.load(open('models/dictionary_SIFT.dat', 'rb'))

index = 0
folders = ['testImages/illustrations', 'testImages/photos']
for folder in folders:
    print("Generate features for {}".format(folder))
    for root, dirs, files in os.walk(folder):
        for dir in dirs:
            print(dir)
            files = [f for f in os.listdir(os.path.join(folder, dir))]
            for file in files:
                filename = os.path.join(folder, dir, file)
                img = cv2.imread(filename)

                print('Prediction for {}'.format(filename))

                glcmFeature = preprocessing.normalize([featureExtraction.glcm(img)])
                hsvHist = preprocessing.normalize([featureExtraction.colorHSVHistogram(img)])
                lbp = preprocessing.normalize([featureExtraction.binaryPattern(img)])

                valuePredicted = modelRfGlcm.predict_proba(glcmFeature)
                print("GLCM {}".format(valuePredicted))

                valuePredicted = modelRfHsv.predict_proba(hsvHist)
                print("HSV Histogram {}".format(valuePredicted))

                valuePredicted = modelRfLBP.predict_proba(lbp)
                print("LBP {}".format(valuePredicted))

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                (kps, descs) = detector.detectAndCompute(gray, None)

                words, distance = vq(descs, dictionarySVMBowSift)
                feature = np.zeros(dictionarySVMBowSift.shape[0], "int32")
                for w in words:
                    feature[w] += 1
                hist = feature / sum(feature)
                valuePredicted = modelSVMBowSift.predict_proba([hist])
                print("BOW SIFT {}".format(valuePredicted))

                imageBuffer = skimage.data.imread(filename)
                image = skimage.transform.resize(imageBuffer, (128, 128))
                image = skimage.color.rgb2gray(image).reshape(1, 128, 128, 1) / 255
                valuePredicted = modelCNN.predict(image)
                print("CNN {}".format(valuePredicted))
