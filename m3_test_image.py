#!/bin/sh

import sys
import cv2
import vehicle_detection2
from extract_features import HogFeatureExtractor
import json

from sklearn.externals import joblib


class Env:
    pass


class App:
    def __init__(self, vehicle_detector):
        self.w = 'preview'
        # opencv window resizing is brain dead
        cv2.namedWindow(self.w)
        cv2.waitKey(33)
        cv2.destroyAllWindows()
        cv2.namedWindow(self.w)
        self.vehicle_detector = vehicle_detector

    def process_image(self, img):
        processed = self.vehicle_detector.process_image(img)
        scaled = cv2.resize(
                processed,
                (1024, int(img.shape[0]*(1024./img.shape[1]))))
        cv2.imshow(self.w, scaled)
        cv2.waitKey(-1)
        return processed 

env_fname, clf_fname, img_fname, out_fname = sys.argv[1:]

with open(env_fname) as f:
    env = json.load(f)
clf = joblib.load(clf_fname)
hog = HogFeatureExtractor(
        cspace=env['colorspace'],
        orient=env['orient'],
        pix_per_cell=env['pix_per_cell'],
        cell_per_block=env['cell_per_block'],
        hog_channel=env['hog_channel'])
vehicle_detector = vehicle_detection2.VehicleDetector(clf, hog)
app = App(vehicle_detector)

img = cv2.imread(img_fname)
output = app.process_image(img)
cv2.imwrite(out_fname, output)
