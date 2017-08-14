import numpy as np
from werkzeug import secure_filename
from src.utils.ML_Tool_Utils import Utils

class PCA(object):

    def __init__(self, numpyFile, d):
        self.data = self.initialiseData(numpyFile)
        self.d = d
        self.utils = Utils()

    def initialiseData(self, numpyFile):
        inputFile = secure_filename(numpyFile.filename)
        numpyFile.save(inputFile)
        return np.load(open(inputFile, 'rb'))

    def save(self):
        filename = self.utils.generateFileName('.npy')
        np.save(open(filename, 'wb'), self.projected)
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
