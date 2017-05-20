#!/usr/bin/env python

from skimage.feature import hog as skhog
import numpy as np
import cv2
import sys
import glob
import os

def hog( infname, outfname):
    img = cv2.imread(infname)
    luv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    hog_arr = []
    for ch in range(3):
        features = skhog(luv[:,:,ch], orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), feature_vector=False)
        assert(np.isfinite(features).all())
        hog_arr.append(features)
    np.save(outfname, np.asarray(hog_arr))

def find_samples( subdir ):
    return sorted(glob.glob(os.sep.join([subdir, '**', '*.jpg']), recursive=True))

if __name__=="__main__":
    assert(len(sys.argv)>2)
    image_dir = sys.argv[1]
    samples = find_samples( image_dir )
    print("Found {} samples".format(len(samples)))
    for infname in samples:
        outfname = os.path.sep.join([sys.argv[2], infname[(len(image_dir)+1):].replace(os.path.sep,'__')]) + '.npy'
        print("hog {} {}".format(infname, outfname))
        hog(infname, outfname)
    print("OK")
