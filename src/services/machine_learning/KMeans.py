import numpy as np
from src.utils.MLToolUtils import Utils
from src import constants

class KMeans(object):

    def __init__(self, numpyFile, k):
        self.utils = Utils()
        self.data = self.utils.initialiseData(numpyFile)
        self.k = k
        self.clusters = None

    def save(self):
        filename = self.utils.generateFileName(constants.numpy_extension)
        np.save(open(filename, constants.file_write_flag), self.clusters)
        return filename

    def compute(self):
        clusterAssignmentList = []
        distortions = []
        for ii in range(10):

            # ansatz of centroid locations
            centroids = self.data[np.random.randint(0, self.data.shape[0], self.k), :]

            eps = 10000
            while eps > 0.001:
                clusterAssignments = np.zeros((self.data.shape[0]))

                for i in range(self.data.shape[0]):
                    currentSample = self.data[i, :]

                    # find closest centroid to sample
                    clusterAssignments[i] = np.linalg.norm(centroids - currentSample, axis=1).argmin()

                # update centroid locations
                centroidsPrev = centroids.copy()
                for j in range(self.k):
                    centroids[j, :] = self.data[np.where(clusterAssignments == j), :].mean(1)[0, :]
                centroidsNew = centroids.copy()

                # threshold for terminating algorithm
                eps = np.linalg.norm(centroidsPrev - centroidsNew)

            distortion = 0
            for i in range(self.k):
                centroid = centroids[i, :]
                samples = self.data[np.where(clusterAssignments == i), :][0, :, :]
                distortion += np.sum(np.linalg.norm(samples - centroid, axis=1))

            clusterAssignmentList.append(clusterAssignments)
            distortions.append(distortion)
        clusterAssignmentList = np.array(clusterAssignmentList)
        distortions = np.array(distortions)
        self.clusters = np.array(clusterAssignmentList[distortions.argmin()])
        return self.save()
