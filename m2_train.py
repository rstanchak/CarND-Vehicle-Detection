#!/usr/bin/env python

import os
import sys
import glob
import numpy as np
import numpy.random as nprand
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
import sklearn.svm as svm

def usage(progname):
    print("USAGE: %s <data directory>" % progname)

if __name__=='__main__':
    assert(len(sys.argv)>=1)
    fileset = glob.glob(os.path.sep.join([sys.argv[1], '*.npy']))
    features_array=[]
    labels_array=[]
    print("Loading {} samples...".format(len(fileset)))
    N = 5000
    idx = list(range( len(fileset) ))
    nprand.shuffle(idx)
    #idx = idx[:N]
    for fname in map(lambda x: fileset[x], idx):
        features = np.load(fname)
        features_array.append( features.ravel() )
        labels_array.append( os.path.basename(fname).split('__')[0] )
    
    X = np.asarray( features_array, dtype=np.float64 )
    labels_idx = dict( map( reversed, enumerate( set(labels_array))))
    y = np.asarray( list(map( labels_idx.get, labels_array ) ))

    print("Training")
    cv = 2
    clf = make_pipeline( StandardScaler(), svm.SVC(C=1,verbose=True))
    scores = cross_val_score(clf, X, y, cv=cv)
    print(scores)
    joblib.dump( clf, 'P5-clf.pk' )

