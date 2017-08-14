from werkzeug import secure_filename
from flask import current_app as app
import os
import cv2
import numpy as np
from skimage.feature import peak_local_max
from scipy.ndimage.filters import convolve
from src.utils.ML_Tool_Utils import Utils

class Harris(object):

    def __init__(self, imageFile):
        self.f = self.initialiseImage(imageFile)
        self.utils = Utils()

        app.logger.info('Initialising hyperparameters...')
        self.w = 3
        self.s = 0.7
        self.k = 0.25
        self.t = 0.1
        self.sigma_l = 2

    def initialiseImage(self, imageFile):
        app.logger.info('Obtaining image...')
        inputFile = secure_filename(imageFile.filename)
        imageFile.save(inputFile)
        input_image = cv2.imread(inputFile, 0) / 255.0
        os.remove(inputFile)
        return input_image

    def createGaussianKernel(self, sigma):
        gx = cv2.getGaussianKernel(self.w, sigma)
        gy = cv2.getGaussianKernel(self.w, sigma)
        return np.sqrt(np.outer(gx, gy))

    def computeScaleSpace(self):
        L = convolve(self.f, self.createGaussianKernel(self.sigma_l))
        Lx = np.gradient(L, axis=0)
        Ly = np.gradient(L, axis=1)
        return Lx, Ly

    def computeResposneMatrices(self, Lx, Ly):
        sigma_i = self.s * self.sigma_l
        g = self.createGaussianKernel(sigma_i)
        A = convolve(Lx * Lx, g)
        B = convolve(Ly * Ly, g)
        C = convolve(Lx * Ly, g)
        return A, B, C

    def findPoints(self, H):
        localMaximaIndices = peak_local_max(H)
        max_local = np.zeros_like(H)
        for i in range(localMaximaIndices.shape[0]):
            x = localMaximaIndices[i, 0]
            y = localMaximaIndices[i, 1]
            max_local[x, y] = H[x, y]
        threshold = 0.1 * np.max(max_local)
        xx, yy = np.where(max_local > threshold)
        return xx, yy

    def draw(self):
        frame = cv2.cvtColor((self.f * 255.0).astype('uint8'), cv2.COLOR_GRAY2BGR)
        for i in range(len(self.x)):
            xxx = self.x[i]
            yyy = self.y[i]
            cv2.circle(frame, (yyy, xxx), 2, (0, 0, 255), 1)
        self.frame = frame

    def save(self):
        filename = self.utils.generateFileName('.jpg')
        cv2.imwrite(filename, self.frame)
        return filename

    def compute(self):
        Lx, Ly = self.computeScaleSpace()
        A, B, C = self.computeResposneMatrices(Lx, Ly)
        H = A * B - self.k * C * C
        self.x, self.y = self.findPoints(H)
        self.draw()
        return self.save()


