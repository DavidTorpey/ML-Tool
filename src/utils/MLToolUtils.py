from uuid import uuid4
import os
from src import constants

class Utils(object):

    def __init__(self):
        pass

    def generateFileName(self, extension):
        return str(uuid4()).replace(constants.dash, constants.empty_string) + extension

    def removeFile(self, file_path):
        os.remove(file_path)