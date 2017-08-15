from uuid import uuid4
import os
from src import constants
import numpy as np
from werkzeug import secure_filename

class Utils(object):

    def __init__(self):
        pass

    def generateFileName(self, extension):
        return str(uuid4()).replace(constants.dash, constants.empty_string) + extension

    def removeFile(self, file_path):
        os.remove(file_path)

    def initialiseData(self, numpyFile):
        inputFile = secure_filename(numpyFile.filename)
        numpyFile.save(inputFile)
        return np.load(open(inputFile, constants.file_read_flag))