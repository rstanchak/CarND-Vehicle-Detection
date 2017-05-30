#!/bin/sh

from moviepy.editor import VideoFileClip
import sys
import cv2
from vehicle_detection import slide_window

class Shape:
    def __init__(self, shape):
        self.shape = shape
class App:
    def __init__(self):
        self.w = 'preview'
        # opencv window resizing is brain dead
        cv2.namedWindow(self.w)
        cv2.waitKey(33)
        cv2.destroyAllWindows()
        cv2.namedWindow(self.w)

    def process_image(self, img):
        shape = Shape(img.shape)
        y0 = int(5*img.shape[0]/9)
        y1 = int(3*img.shape[0]/4)
        x0 = int(img.shape[1]/5)
        x1 = int(4*img.shape[1]/5)
        count=0

        for win in slide_window(shape, x_start_stop=(x0,x1), y_start_stop=(y0,y1), xy_overlap=(0.9,0.9)):
            cv2.rectangle(img, win[0], win[1], (0,255,0), 1)
            count+=1


        #y0 = int(11*img.shape[0]/16)
        y0 = int(img.shape[0]/2)
        #y1 = int(7*img.shape[0]/8)
        y1 = int(3*img.shape[0]/4)
        x0 = 0
        x1 = img.shape[1]

        for win in slide_window(shape, xy_window=(128,128), x_start_stop=(x0,x1), y_start_stop=(y0,y1), xy_overlap=(0.9,0.9)):
            cv2.rectangle(img, win[0], win[1], (255,0,0), 1)
            count+=1

        #y0 = int(3*img.shape[0]/5)
        #y0 = int(5*img.shape[0]/9)
        y1 = img.shape[0]
        for win in slide_window(shape, xy_window=(256,256), x_start_stop=(x0,x1), y_start_stop=(y0,y1), xy_overlap=(0.9,0.9)):
            cv2.rectangle(img, win[0], win[1], (0,0,255), 1)
            count+=1

        print(count)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow(self.w, rgb)
        cv2.waitKey(33)
        return img

input_fname = sys.argv[1]

app = App( )
input_clip = VideoFileClip( input_fname )
output_clip = input_clip.fl_image( app.process_image )
output_clip.write_videofile("test.mp4", audio=False)
