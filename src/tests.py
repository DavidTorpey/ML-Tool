import unittest
from app import app
from src.utils.ML_Tool_Utils import Utils

class TestFlaskApiUsingRequests(unittest.TestCase):

    def setUp(self):
        app.config['TESTING'] = True
        self.app = app.test_client()
        self.utils = Utils()

    def test_pca(self):
        response = self.app.post('/MachineLearning/PCA', data={'file':open('../iris.npy', 'rb')})
        self.utils.removeFile('iris.npy')
        self.assertEqual(response.status_code, 200)

    def test_harris(self):
        response = self.app.post('/ComputerVision/HarrisDetection', data={'file':open('../lena.jpg', 'rb')})
        self.assertEqual(response.status_code, 200)

if __name__ == "__main__":
    unittest.main()