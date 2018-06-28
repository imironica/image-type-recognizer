# Code source by Ionuț Mironică
# This scripts tests the performance of the features using late fusion techniques:
#  - mean of the best algorithms
#  - max of the algorithms
#  - hard voting strategy

import os
import cv2
import numpy as np
import pandas as pd
from os import listdir
from os.path import join
from scipy.cluster.vq import *
from sklearn import preprocessing
from skimage.feature import local_binary_pattern
from skimage.feature import greycomatrix, greycoprops
from FeatureExtraction import FeatureExtraction

class DatabaseFeatureExtraction(object):
    """Extract classical image features for the entire database

    document type classification (GLCM, LBP, BOW)

        Keyword arguments:
        gray -- the grayscale image matrix
        descriptorType -- the imaginary part (default SIFT / possible values SIFT / SURF / ORB)
        """

    def __init__(self):
        self.folderPhoto = 'photo'
        self.folderClipArt = `
        self.featuresExtraction = FeatureExtraction()

    def getFunction(self, featureType):
        if featureType == 'LBP':
            return self.featuresExtraction.binaryPattern
        if featureType == 'GLCM':
            return self.featuresExtraction.glcm
        if featureType == 'RGB':
            return self.featuresExtraction.colorRGBHistogram
        if featureType == 'HSV':
            return self.featuresExtraction.colorHSVHistogram

        return self.binaryPattern

    def computeKeypoints(self, image, descriptorType):
        """Compute the keypoints descriptions and locations for a grayscale image.

            Keyword arguments:
            gray -- the grayscale image matrix
            descriptorType -- default SIFT / possible values SIFT / SURF / ORB
            """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        descs = None
        if descriptorType == "SIFT":
            detector = cv2.xfeatures2d.SIFT_create()
            (kps, descs) = detector.detectAndCompute(gray, None)

        if descriptorType == "SURF":
            detector = cv2.xfeatures2d.SURF_create()
            (kps, descs) = detector.detectAndCompute(gray, None)

        if descriptorType == "ORB":
            detector = cv2.ORB_create()
            (kps, descs) = detector.detectAndCompute(gray, None)

        if descs is None:
            return (None, None)
        return (kps, descs.astype("float"))

    def binaryPattern(self, image):
        """Compute the LBP features  for a grayscale image.

            Keyword arguments:
            gray -- the grayscale image matrix
            """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray, 24, 8, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, 24 + 3),
                                 range=(0, 24 + 2))
        return hist.tolist()

    def colorRGBHistogram(self, image):
        """Compute the Color features  for a RGB image.

            Keyword arguments:
            gray -- the grayscale image matrix
            """

        hist = cv2.calcHist([image], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256]).flatten()
        return hist

    def colorHSVHistogram(self, image):
        """Compute the Color features  for a RGB image.

            Keyword arguments:
            gray -- the grayscale image matrix
            """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [4, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
        return hist

    def glcm(self, gray):
        """Compute the  Gray level Co-occurrence matrix features (GLCM) for a grayscale image.

            Keyword arguments:
            gray -- the grayscale image matrix
            """
        distances = [1, 2, 3]
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        properties = ['energy', 'homogeneity']
        glcm = greycomatrix(gray,
                            distances=distances,
                            angles=angles,
                            symmetric=True,
                            normed=True)

        hist = np.hstack([greycoprops(glcm, prop).ravel() for prop in properties]).tolist()
        return hist

    def getFunction(self, featureType):
        if featureType == 'LBP':
            return self.binaryPattern
        if featureType == 'GLCM':
            return self.glcm
        if featureType == 'RGB':
            return self.colorRGBHistogram
        if featureType == 'HSV':
            return self.colorHSVHistogram

        return self.binaryPattern

    def generateFeatures(self, featureType):
        """Compute descriptions for a image.

            Keyword arguments:
            featureType -- default LBP / possible values LBP / GLCM / RGB / HSV)
            """
        features = []
        labels = []
        functionFeature = self.getFunction(featureType)

        folders = [self.folderPhoto, self.folderClipArt]

        index = 0
        for folder in folders:
            print("Generate features for {}".format(folder))
            for root, dirs, files in os.walk(folder):
                for dir in dirs:
                    print(dir)
                    files = [f for f in listdir(join(folder, dir))]
                    for file in files:
                        img = cv2.imread(join(folder, dir, file))
                        if img is not None:
                            feature = functionFeature(img)
                            labels.append(index)
                            features.append(feature)
            index += 1
        print(len(features))
        features = preprocessing.normalize(features)

        df = pd.DataFrame(features)
        df.to_csv("{}.csv".format(featureType), index=False)

        df = pd.DataFrame(labels)
        df.to_csv("labels.csv", index=False)

    def generateBOW(self, descriptorType, dictionarySize):
        """Compute descriptions for a grayscale image.

            Keyword arguments:
            featureType -- default LBP / possible values LBP / GLCM)
            """
        dictionaryList = []
        percentDictionary = 25

        # Take the images for the dictionary
        print('Acquiring the dictionary keypoints')
        folders = [self.folderPhoto, self.folderClipArt]

        for folder in folders:
            for root, dirs, files in os.walk(folder):
                for dir in dirs:
                    print(dir)
                    files = [f for f in listdir(join(folder, dir))]
                    index = 0
                    for file in files:
                        index += 1
                        if index % percentDictionary == 0:
                            img = cv2.imread(join(folder, dir, file))
                            if img is not None:
                                (kps, descs) = self.computeKeypoints(img, descriptorType)
                                if len(dictionaryList) == 0:
                                    dictionaryList = descs
                                dictionaryList = np.vstack((dictionaryList, descs))

        dictionaryList = dictionaryList[1::20]

        print("Perform KMEANS clustering ..")
        dictionary, variance = kmeans(dictionaryList, dictionarySize, 1)
        import pickle
        pickle.dump(dictionary, open('models/dictionary_{}.dat'.format(descriptorType),'wb'))
        print("Generate BOW features ..")
        features = []
        labels = []
        for root, dirs, files in os.walk(self.folderPhoto):
            for dir in dirs:
                print(dir)
                files = [f for f in listdir(join(self.folderPhoto, dir))]
                for file in files:
                    img = cv2.imread(join(self.folderPhoto, dir, file))
                    if img is not None:

                        (kps, descs) = self.computeKeypoints(img, descriptorType)
                        if descs is not None:
                            words, distance = vq(descs, dictionary)
                            feature = np.zeros(dictionary.shape[0], "int32")
                            for w in words:
                                feature[w] += 1
                            hist = feature / sum(feature)

                            labels.append(0)
                            features.append(hist)

        for root, dirs, files in os.walk(self.folderClipArt):
            for dir in dirs:
                print(dir)
                files = [f for f in listdir(join(self.folderClipArt, dir))]
                for file in files:
                    img = cv2.imread(join(self.folderClipArt, dir, file))
                    if img is not None:
                        (kps, descs) = self.computeKeypoints(img, descriptorType)
                        if descs is not None:
                            words, distance = vq(descs, dictionary)
                            feature = np.zeros(dictionary.shape[0], "int32")
                            for w in words:
                                feature[w] += 1
                            hist = feature / sum(feature)

                            labels.append(1)
                            features.append(hist)

        df = pd.DataFrame(features)
        df.to_csv("{}_{}.csv".format(descriptorType, dictionarySize), index=False)

        df = pd.DataFrame(labels)
        df.to_csv("labels.csv", index=False)