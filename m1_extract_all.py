#!/usr/bin/env python

import numpy as np
import cv2
import sys
import glob
import os
import json
from extract_features import FeatureExtractor

def find_samples( subdir ):
    return sorted(glob.glob(os.sep.join([subdir, '**', '*.png']), recursive=True) 
            + glob.glob(os.sep.join([subdir, '**', '*.jpg']), recursive=True))

def usage():
    print ("USAGE: {} <env fname> <input directory> <output directory>")

if __name__=="__main__":
    assert(len(sys.argv)>2)
    with open(sys.argv[1]) as f:
        env = json.load(f)
    image_dir = sys.argv[2]
    f = FeatureExtractor( spsize=env['spatial_size'],
            histbins=env['histbins'],
            cspace=env['colorspace'],
            hog_orient=env['orient'],
            hog_pix_per_cell=env['pix_per_cell'],
            hog_cell_per_block=env['cell_per_block'] )
    samples = find_samples( image_dir )
    print("Found {} samples".format(len(samples)))
    for infname in samples:
        outfname = os.path.sep.join([sys.argv[3], infname[(len(image_dir)+1):].replace(os.path.sep,'__')]) + '.npy'
        print("features {} {} ({})".format(infname, outfname, json.dumps(env)))
        img = cv2.imread(infname)
        f.preprocess(img, env['train_image_size'])
        features = f.extract((0,0))
        np.save(outfname, features)
    print("OK")
