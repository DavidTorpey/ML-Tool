from src.services.computer_vision.Harris import Harris
from flask import current_app as app

class MainService(object):

    def __init__(self):
        pass

    def detectHarrisPoints(self, image):
        self.harris_detector = Harris(image)

        app.logger.info("Computing Harris points...")
        return self.harris_detector.compute()

