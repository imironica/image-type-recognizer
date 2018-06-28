# Code source by Ionuț Mironică
# This script performs a set of experiments using a deep learning architecture

import os
import numpy as np
import skimage.data
import skimage.transform
from sklearn.metrics import confusion_matrix, accuracy_score
from os.path import join, isfile

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dropout, Flatten
from keras.layers import Dense, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam
imageSize = 128


def loadData(dataDir, resize=False, label=0, size=(imageSize, imageSize)):
    """Loads a data set and returns two lists:
    images: a list of Numpy arrays, each representing an image.
    labels: a list of numbers that represent the images labels.
    """
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(dataDir)
                   if os.path.isdir(os.path.join(dataDir, d))]
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(dataDir, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir) if f.endswith(".jpg")]
        # For each label, load it's images and add them to the images list.
        # And add the label number (i.e. directory name) to the labels list.
        for f in file_names:
            try:
                if resize:
                    imageBuffer = skimage.data.imread(f)
                    image = skimage.transform.resize(imageBuffer, size)
                    image = skimage.color.rgb2gray(image)
                    images.append(image)
                labels.append(label)
            except Exception as err:
                print(f)
                pass
    return images, labels


def readDatabase(size=(imageSize, imageSize, 1)):
    print('Reading dataset ...')

    if os.path.exists('images.dat'):
        images = np.load('images_gray.dat')
        labels = np.load('labels_gray.dat')
    else:
        xPhoto, yPhoto = loadData("photo", resize=True, size=size, label=0)
        xClipArt, yClipArt = loadData("clip-art", resize=True, size=size, label=1)
        xPhoto = np.array(xPhoto)
        xClipArt = np.array(xClipArt)
        images = np.concatenate((xPhoto, xClipArt), axis=0)
        labels = np.concatenate((yPhoto, yClipArt), axis=0)

        images.dump('images_gray.dat')
        labels.dump('labels_gray.dat')

    labelsTest = labels[1::2]
    # Preprocess the training data
    labelsCount = len(set(labels))
    labels = to_categorical(labels, num_classes=labelsCount)

    # Scale between 0 and 1
    images = np.array(images)

    images = np.reshape(images, (images.shape[0], imageSize, imageSize, 1))
    images = images / 255.0

    xTrain = images[0::2, :]
    yTrain = labels[0::2]
    xTest = images[1::2, :]
    yTest = labels[1::2]
    return xTrain, yTrain, xTest, yTest, labelsTest


def showConfusionMatrix(yLabels, predictedValues):
    predictedLabels = np.argmax(predictedValues, axis=1)

    accuracy = accuracy_score(y_true=yLabels, y_pred=predictedLabels)
    matrix = confusion_matrix(y_true=yLabels, y_pred=predictedLabels)
    print(accuracy)
    print(matrix)


xTrain, yTrain, xTest, yTest, labelsTest = readDatabase()

# Network parameters
firstConvLayerDepth = 2
numberOfNeurons = 128

# Training hyper-parameters
learningRate = 0.001
noOfEpochs = 10
batchSize = 32

numberOfClasses = yTrain.shape[1]
featureSize = xTrain.shape[1]


# Network parameters
firstConvLayerDepth = 4
secondConvLayerDepth = 8

numberOfNeurons = 256
dropoutPerLayer = 0.10
kernelSize = (5, 5)

model = Sequential()
model.add(Conv2D(firstConvLayerDepth, kernel_size=kernelSize,
                 activation='relu',
                 strides=(1, 1),
                 padding='same',
                 input_shape=(imageSize, imageSize, 1)))

model.add(BatchNormalization())
model.add(Dropout(dropoutPerLayer))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(secondConvLayerDepth, kernel_size=kernelSize,
                 activation='relu',
                 strides=(1, 1),
                 padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(dropoutPerLayer))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())

model.add(Dense(numberOfNeurons, activation='relu'))
model.add(Dropout(dropoutPerLayer))
model.add(Dense(numberOfClasses, activation='softmax'))

sgd = Adam()
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x=xTrain,
          y=yTrain,
          epochs=noOfEpochs,
          batch_size=batchSize,
          verbose=1)

predictedValues = model.predict(xTest, batch_size=1)

print(predictedValues)
showConfusionMatrix(labelsTest, predictedValues)

# Save the dictionary model
modelFilename = 'models/conv_model_{}_{}_{}_{}.dat'.format(firstConvLayerDepth, secondConvLayerDepth, numberOfNeurons, dropoutPerLayer)
model.save(modelFilename)
print("Model saved")

