from flask import Flask
from flask import request
from flask import send_file
import os
from src.services.MainService import MainService
from src import constants

app = Flask(__name__)

mainService = MainService()

@app.route(constants.harris_endpoint, methods=[constants.post])
def detectHarrisPoints():
    app.logger.info("Request to detect Harris points...")
    image = request.files[constants.payload_name]

    image_path = mainService.detectHarrisPoints(image)
    resultant_image = send_file(image_path, mimetype=constants.jpg_mimetype)
    os.remove(image_path)

    return resultant_image

@app.route(constants.pca_endpoint, methods=[constants.post])
def performPCA():
    app.logger.info("Request to perform PCA...")
    numpy_data = request.files[constants.payload_name]
    dim = int(request.args.get(constants.dim_param))

    numpy_file_path = mainService.performPCA(numpy_data, dim)
    projected_data = send_file(numpy_file_path)
    os.remove(numpy_file_path)

    return projected_data

@app.route(constants.kmeans_endpoint, methods=[constants.post])
def performKMeans():
    app.logger.info("Request to perform k-Means...")
    numpy_data = request.files[constants.payload_name]
    k = int(request.args.get(constants.num_clusters))

    numpy_file_path = mainService.performKMeans(numpy_data, k)
    clusterLabels = send_file(numpy_file_path)
    os.remove(numpy_file_path)

    return clusterLabels


if __name__ == '__main__':
    app.run(debug=True, port=constants.port, host="0.0.0.0", threaded=True)