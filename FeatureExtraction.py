# Code source by Ionuț Mironică
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.feature import greycomatrix, greycoprops


class FeatureExtraction(object):
    """Extract classical image features for document type classification (GLCM, LBP, BOW)
    """

    def __init__(self):
        pass

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

    def glcm(self, image):
        """Compute the  Gray level Co-occurrence matrix features (GLCM) for a grayscale image.

            Keyword arguments:
            gray -- the grayscale image matrix
            """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

