import numpy as np
from werkzeug import secure_filename
from src.utils.MLToolUtils import Utils
from src import constants

class PCA(object):

    def __init__(self, numpyFile, d):
        self.utils = Utils()
        self.data = self.utils.initialiseData(numpyFile)
        self.d = d

    def save(self):
        filename = self.utils.generateFileName(constants.numpy_extension)
        np.save(open(filename, constants.file_write_flag), self.projected)
        return filename

    def compute(self):
        m, n = self.data.shape
        self.data -= self.data.mean(axis=0)
        R = np.cov(self.data, rowvar=False)
        evals, evecs = np.linalg.eigh(R)
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:, idx]
        evals = evals[idx]
        evecs = evecs[:, :self.d]
        self.projected = np.dot(evecs.T, self.data.T).T
        return self.save()
