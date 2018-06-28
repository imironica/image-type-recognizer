# Code source by Ionuț Mironică
# This script

import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# compute the predictions on random forests
def getScoreRF(featureName, yTrain):

    xTrainFilename = '{}.csv'.format(featureName)
    features = pd.read_csv(xTrainFilename).as_matrix()
    xTrain = features[0::2, :]
    xTest = features[1::2, :]

    # Train the model
    clfRF = RandomForestClassifier(n_estimators=200, criterion="gini")
    clfRF.fit(xTrain, yTrain)

    # Compute the accuracy of the model
    valuePredicted = clfRF.predict_proba(xTest)

    filename = "models/RF_{}.dat".format(featureName)
    pickle.dump(clfRF, open(filename, 'wb'))
    print("{} saved".format(filename))
    return valuePredicted

yTrainFilename = 'labels.csv'
labels = pd.read_csv(yTrainFilename).values
yTest = labels[1::2].ravel()
yTrain = labels[0::2].ravel()


valuesPredictedHist = getScoreRF('HSV', yTrain)
valuesPredictedLBP = getScoreRF('LBP', yTrain)
valuesPredictedGLCM = getScoreRF('GLCM', yTrain)


# Late fusion of aggregated values
valuesPredicted = (valuesPredictedHist + valuesPredictedGLCM + valuesPredictedLBP) / 3
fusedPredictions = []
for value in valuesPredicted:
    if value[0] > 0.5:
        fusedPredictions.append(0)
    else:
        fusedPredictions.append(1)
accuracy = accuracy_score(y_true=yTest, y_pred=fusedPredictions)
print(accuracy)

confusionMatrix = confusion_matrix(y_true=yTest, y_pred=fusedPredictions)
print(confusionMatrix)

# Late fusion using max prediction
valuesPredicted = np.maximum(valuesPredictedHist, valuesPredictedGLCM, valuesPredictedLBP)
fusedPredictions = []
for value in valuesPredicted:
    if value[0] > 0.5:
        fusedPredictions.append(0)
    else:
        fusedPredictions.append(1)
accuracy = accuracy_score(y_true=yTest, y_pred=fusedPredictions)
print(accuracy)

confusionMatrix = confusion_matrix(y_true=yTest, y_pred=fusedPredictions)
print(confusionMatrix)

# Late fusion using Majority vote prediction
fusedPredictions = []
for (valueHist, valueGLCM, valuesPredictedLBP) in zip(valuesPredictedHist, valuesPredictedGLCM, valuesPredictedLBP):
    value = 0
    if valuesPredictedLBP[0] < 0.5:
        value += 1
    if valueGLCM[0] < 0.5:
        value += 1
    if valueHist[0] < 0.5:
        value += 1

    if value >= 2:
        fusedPredictions.append(1)
    else:
        fusedPredictions.append(0)
accuracy = accuracy_score(y_true=yTest, y_pred=fusedPredictions)
print(accuracy)

confusionMatrix = confusion_matrix(y_true=yTest, y_pred=fusedPredictions)
print(confusionMatrix)
