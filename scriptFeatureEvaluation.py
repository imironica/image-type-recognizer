# Code source by Ionuț Mironică
# This scripts tests the performance of the features with several
# state-of-the-art machine learning algorithms

from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
import argparse
import pandas as pd

# Set the arguments for the script
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--featureName", required=False,
                help="Feature name (e.g., HSV, RGB, LBP, SIFT_1024, GLCM, SURF_1024)")
ap.add_argument("-l", "--labels", required=False,
                help="Labels filename")
args = vars(ap.parse_args())

featureName = 'HSV'
if args["featureName"] is not None:
    featureName = int(args["featureName"])

yTrainFilename = 'labels.csv'
if args["labels"] is not None:
    yTrainFilename = int(args["labels"])

# Set to run some specfic ML algorithms
computeSVM = True
computeSGD = True
computeAdaboost = True
computeGradientBoosting = True
computeRandomForest = True
computeExtremellyRandomForest = True
computeSVMRBF = True

# Read the files that contains the computed features
xTrainFilename = '{}.csv'.format(featureName)

features = pd.read_csv(xTrainFilename).as_matrix()
labels = pd.read_csv(yTrainFilename).values

# Split the features in two parts (train set and test set)
xTrain = features[0::2, :]
yTrain = labels[0::2].ravel()
xTest = features[1::2, :]
yTest = labels[1::2].ravel()


# =================================================================================================#
# Stochastic gradient model
# Train the model
if computeSGD:
    descriptorName = 'SGD'
    print("Train {}".format(descriptorName))
    clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5, tol=None)
    clf.fit(xTrain, yTrain)

    # Compute the accuracy of the model
    valuePredicted = clf.predict(xTest)
    accuracy = accuracy_score(y_true=yTest, y_pred=valuePredicted)
    confusionMatrix = confusion_matrix(y_true=yTest, y_pred=valuePredicted)
    print('{}: {}'.format(descriptorName, accuracy))
    print(confusionMatrix)
# =================================================================================================#

# AdaBoost model
if computeAdaboost:
    descriptorName = 'Adaboost Classifier '
    print("Train {}".format(descriptorName))
    clf = AdaBoostClassifier(n_estimators=100)
    clf.fit(xTrain, yTrain)

    # Compute the accuracy of the model
    valuePredicted = clf.predict(xTest)
    accuracy = accuracy_score(y_true=yTest, y_pred=valuePredicted)
    confusionMatrix = confusion_matrix(y_true=yTest, y_pred=valuePredicted)
    print('{}: {}'.format(descriptorName, accuracy))
    print(confusionMatrix)

# =================================================================================================#
# Gradient Boosting Classifier
if computeGradientBoosting:
    descriptorName = 'Gradient Boosting Classifier'
    print("Train {}".format(descriptorName))
    clf = GradientBoostingClassifier(n_estimators=200, learning_rate=1.0, max_depth=1, random_state=0)
    clf.fit(xTrain, yTrain)

    # Compute the accuracy of the model
    valuePredicted = clf.predict(xTest)
    accuracy = accuracy_score(y_true=yTest, y_pred=valuePredicted)
    confusionMatrix = confusion_matrix(y_true=yTest, y_pred=valuePredicted)
    print('{}: {}'.format(descriptorName, accuracy))
    print(confusionMatrix)

# =================================================================================================#

# Random Forest Classifier
if computeRandomForest:
    descriptorName = 'Random Forest Classifier'
    print("Train {}".format(descriptorName))
    # Train the model
    clfRF = RandomForestClassifier(n_estimators=200, criterion="gini")
    clfRF.fit(xTrain, yTrain)

    # Compute the accuracy of the model
    valuePredicted = clfRF.predict(xTest)
    accuracy = accuracy_score(y_true=yTest, y_pred=valuePredicted)
    confusionMatrix = confusion_matrix(y_true=yTest, y_pred=valuePredicted)
    print('{}: {}'.format(descriptorName, accuracy))
    print(confusionMatrix)

# Extremely RandomForest Classifier
if computeExtremellyRandomForest:
    descriptorName = 'Extremelly Trees Classifier'
    print("Train {}".format(descriptorName))
    # Train the model
    clfRF = ExtraTreesClassifier(n_estimators=200, criterion="gini")
    clfRF.fit(xTrain, yTrain)

    # Compute the accuracy of the model
    valuePredicted = clfRF.predict(xTest)
    accuracy = accuracy_score(y_true=yTest, y_pred=valuePredicted)
    confusionMatrix = confusion_matrix(y_true=yTest, y_pred=valuePredicted)
    print('{}: {}'.format(descriptorName, accuracy))
    print(confusionMatrix)

# Support vector machines (Linear and RBF kernel)
descriptorName = 'SVM Linear'
cValues = [0.001, 0.01, 0.1, 1, 10, 100, 500, 1000]
if computeSVM:
    for cValue in cValues:
        descriptorName = 'Linear SVM with C={} '.format(cValue)
        print("Train {}".format(descriptorName))
        clfSVM = svm.SVC(C=cValue, kernel='linear', verbose=False)
        clfSVM.fit(xTrain, yTrain)

        # Compute the accuracy of the model
        valuePredicted = clfSVM.predict(xTest)
        accuracy = accuracy_score(y_true=yTest, y_pred=valuePredicted)
        confusionMatrix = confusion_matrix(y_true=yTest, y_pred=valuePredicted)
        print('{}: {}'.format(descriptorName, accuracy))
        print(confusionMatrix)

descriptorName = 'SVM RBF'
cValues = [0.001, 0.01, 0.1, 1, 10, 100, 500, 1000]
if computeSVMRBF:
    for cValue in cValues:
        descriptorName = 'SVM with C={} '.format(cValue)
        print("Train {}".format(descriptorName))
        clfSVM = svm.SVC(C=cValue, class_weight=None,
                         gamma='auto', kernel='rbf',
                         verbose=False)
        clfSVM.fit(xTrain, yTrain)

        # Compute the accuracy of the model
        valuePredicted = clfSVM.predict(xTest)
        accuracy = accuracy_score(y_true=yTest, y_pred=valuePredicted)
        confusionMatrix = confusion_matrix(y_true=yTest, y_pred=valuePredicted)
        print('{}: {}'.format(descriptorName, accuracy))
        print(confusionMatrix)
