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

if __name__ == '__main__':
    app.run(debug=True, port=4000, host="0.0.0.0", threaded=True)