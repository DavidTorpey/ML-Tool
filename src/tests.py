import unittest
from app import app
from src.utils.MLToolUtils import Utils
from src import constants

class TestMLTool(unittest.TestCase):

    def setUp(self):
        app.config[constants.testing_property] = True
        self.app = app.test_client()
        self.utils = Utils()

    def test_pca(self):
        dim = "2"
        response = self.app.post(constants.pca_endpoint + '?' + constants.dim_param + '=' + dim, data={constants.payload_name:open('../iris.npy', constants.file_read_flag)})
        self.utils.removeFile('iris.npy')
        self.assertEqual(response.status_code, 200)

    def test_harris(self):
        response = self.app.post(constants.harris_endpoint, data={constants.payload_name:open('../lena.jpg', constants.file_read_flag)})
        self.assertEqual(response.status_code, 200)

    def test_kmeans(self):
        k = "2"
        response = self.app.post(constants.kmeans_endpoint + '?' + constants.num_clusters + '=' + k, data={constants.payload_name: open('../iris.npy', constants.file_read_flag)})
        self.utils.removeFile('iris.npy')
        self.assertEqual(response.status_code, 200)

if __name__ == "__main__":
    unittest.main()