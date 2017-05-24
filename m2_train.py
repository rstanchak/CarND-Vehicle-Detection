#!/usr/bin/env python

import os
import sys
import glob
import numpy as np
import numpy.random as nprand
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

from sklearn.externals import joblib
import sklearn.svm as svm
from sklearn.metrics import classification_report

def usage(progname):
    print("USAGE: %s <data directory>" % progname)

if __name__=='__main__':
    assert(len(sys.argv)>=1)
    fileset = glob.glob(os.path.sep.join([sys.argv[1], '*.npy']))
    features_array=[]
    labels_array=[]
    print("Loading {} samples...".format(len(fileset)))
    N = -1
    idx = list(range( len(fileset) ))
    nprand.shuffle(idx)
    idx = idx[:N]
    for fname in map(lambda x: fileset[x], idx):
        features = np.load(fname)
        features_array.append( features.ravel() )
        labels_array.append( os.path.basename(fname).split('__')[0] )
    
    X = np.asarray( features_array, dtype=np.float64 )
    labels_idx = dict( map( reversed, enumerate( set(labels_array))))
    y = np.asarray( list(map( labels_idx.get, labels_array ) ))

    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33)
    print("Training")
    clf = make_pipeline( StandardScaler(), svm.SVC(C=1,verbose=True))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=list(labels_idx.keys())))
    joblib.dump( clf, 'P5-clf.pk' )

