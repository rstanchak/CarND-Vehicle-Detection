#!/bin/sh

from moviepy.editor import VideoFileClip
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
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        processed = self.vehicle_detector.process_image(bgr)
        #scaled = cv2.resize(
        #        processed,
        #        (1024, int(img.shape[0]*(1024./img.shape[1]))))
        cv2.imshow(self.w, processed)
        cv2.waitKey(33)
        return cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

env_fname, clf_fname, input_fname, output_fname = sys.argv[1:]

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
input_clip = VideoFileClip( input_fname )
output_clip = input_clip.fl_image( app.process_image )

output_clip.write_videofile(output_fname, audio=False)
