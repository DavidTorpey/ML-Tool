from uuid import uuid4
import os

class Utils(object):

    def __init__(self):
        pass

    def generateFileName(self, extension):
        return str(uuid4()).replace('-','') + extension

    def removeFile(self, file_path):
        os.remove(file_path)