# Code source by Ionuț Mironică
# this script saves the features used in experiments in csv files

from DatabaseFeatureExtraction import DatabaseFeatureExtraction

if __name__ == '__main__':

    featureExtraction = DatabaseFeatureExtraction()

    print(">> Generate RGB histogram")
    featureExtraction.generateFeatures('RGB')

    print(">> Generate HSV histogram")
    featureExtraction.generateFeatures('HSV')
   
    print(">> Generate LBP features")
    featureExtraction.generateFeatures('LBP')

    print(">> Generate GLCM features")
    featureExtraction.generateFeatures('glcm')

    print(">> Generate BOW-SIFT features")
    featureExtraction.generateBOW("SIFT", 1024)

    print(">> Generate BOW-SURF features")
    featureExtraction.generateBOW("SURF", 1024)

