from flask import current_app as app
from src.services.computer_vision.Harris import Harris
from src.services.machine_learning.PCA import PCA

class MainService(object):

    def __init__(self):
        pass

    def detectHarrisPoints(self, image):
        self.harris_detector = Harris(image)

        app.logger.info("Computing Harris points...")
        return self.harris_detector.compute()

    def performPCA(self, numpyFile):
        self.pca = PCA(numpyFile, 2)

        app.logger.info("Performing principal components analysis on data...")
        return self.pca.compute()