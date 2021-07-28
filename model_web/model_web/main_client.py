import requests
import argparse
import os
import numpy as np
import cv2
# Initialize the PyTorch REST API endpoint URL.
PyTorch_REST_API_URL = 'http://127.0.0.1:5000/predict'

# 传入图片的路径
def predict_result(image_dir):
    # Initialize image path

    # image = open(image_path, 'rb').read()
    payload = {'data': image_dir}

    # Submit the request.
    r = requests.post(PyTorch_REST_API_URL, data=payload).json()
    # F:\bigdata\depath\AEPic_139_1.5x_3
    # Ensure the request was successful.
    if r['success']:
        # Loop over the predictions and display them.
        for (i, result) in enumerate(r['predictions']):
            print('{}. {}: {:.4f}'.format(i + 1, result['label'],
                                          result['probability']))
    # Otherwise, the request failed.
    else:
        print('Request failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classification demo')
    parser.add_argument('--file', type=str, help='test image file')

    args = parser.parse_args()
    predict_result(args.file)
