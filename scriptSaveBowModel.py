# Code source by Ionuț Mironică
# This script saves the BOW model with Linear SVM

from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import pandas as pd

# Set the arguments for the script

featureName = 'SIFT_1024'
yTrainFilename = 'labels.csv'

# Read the files that contains the computed features
xTrainFilename = '{}.csv'.format(featureName)

features = pd.read_csv(xTrainFilename).as_matrix()
labels = pd.read_csv(yTrainFilename).values

# Split the features in two parts (train set and test set)
xTrain = features[0::2, :]
yTrain = labels[0::2].ravel()
xTest = features[1::2, :]
yTest = labels[1::2].ravel()

# Support vector machines (Linear and RBF kernel)
descriptorName = 'SVM Linear'
c = 500

print("Train {}".format(descriptorName))
clfSVM = svm.SVC(C=c, kernel='linear', verbose=False, probability=True)
clfSVM.fit(xTrain, yTrain)

# Compute the accuracy of the model
valuePredicted = clfSVM.predict(xTest)
accuracy = accuracy_score(y_true=yTest, y_pred=valuePredicted)
confusionMatrix = confusion_matrix(y_true=yTest, y_pred=valuePredicted)
print('{}: {}'.format(descriptorName, accuracy))
print(confusionMatrix)

filename = "models/BOW_SIFT_SVM.dat"
pickle.dump(clfSVM, open(filename, 'wb'))
print("{} saved".format(filename))

