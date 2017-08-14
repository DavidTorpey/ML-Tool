from flask import Flask
from flask import request
from flask import send_file
import os
from src.services.MainService import MainService

app = Flask(__name__)

mainService = MainService()

@app.route('/ComputerVision/HarrisDetection', methods=['POST'])
def detectHarrisPoints():
    app.logger.info("Request to detect Harris points...")
    image = request.files['file']

    image_path = mainService.detectHarrisPoints(image)
    resultant_image = send_file(image_path, mimetype='image/jpg')
    os.remove(image_path)

    return resultant_image

@app.route('/MachineLearning/PCA', methods=['POST'])
def performPCA():
    app.logger.info("Request to perform PCA...")
    numpy_data = request.files['file']

    numpy_file_path = mainService.performPCA(numpy_data)
    projected_data = send_file(numpy_file_path)
    os.remove(numpy_file_path)

    return projected_data



if __name__ == '__main__':
    app.run(debug=True, port=4000, host="0.0.0.0", threaded=True)