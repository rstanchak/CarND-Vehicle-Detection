import numpy as np
import cv2
from skimage.feature import hog

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features

def convert_color( img, cspace ):
    # apply color conversion if other than 'RGB'
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        elif cspace == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    else:
        feature_image = np.copy(img)
    return feature_image

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_feature(rgb, cspace='RGB', orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0, feature_vec=True):
    feature_image = convert_color(rgb, cspace)

    # Call get_hog_features() with vis=False, feature_vec=True
    if hog_channel == 'ALL':
        hog_features = []
        for channel in range(feature_image.shape[2]):
            hog_features.append(get_hog_features(feature_image[:,:,channel],
                                orient, pix_per_cell, cell_per_block,
                                vis=False, feature_vec=feature_vec))
        if feature_vec:
            hog_features = np.ravel(hog_features)
        else:
            hog_features = np.asarray(hog_features)
    else:
        hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                    pix_per_cell, cell_per_block, vis=False, feature_vec=feature_vec)
    if not np.isfinite(hog_features).all():
        # sanity check
        raise ValueError
    # Append the new feature vector to the features list
    return hog_features
    
# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace='RGB', orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = cv2.imread(file)
        try:
            features.append(extract_feature(image,cspace,orient,pix_per_cell,cell_per_block,hog_channel))
        except ValueError:
            print(file)
            continue
    return features

def extract_window(img, window):
    return img[window[0][1]:window[1][1],window[0][0]:window[1][0],:]

class FeatureExtractor:
    def __init__(self, cspace='RGB', spsize=16, histbins=8, hog_orient=9, hog_pix_per_cell=8, hog_cell_per_block=2):
        self.hog = HogFeatureExtractor( cspace, hog_orient, hog_pix_per_cell, hog_cell_per_block, 'ALL' )
        self.sp = SpatialFeatureExtractor( spsize )
        self.hist = ColorHistFeatureExtractor( cspace, histbins )

    def preprocess(self, img, window_size):
        self.window_size = window_size
        self.hog.preprocess( img )
        self.sp.preprocess( img, window_size )
        self.hist.preprocess( img )

    def extract(self, p0):
        window = (p0, (p0[0]+self.window_size, p0[1]+self.window_size))
        features_array=[]
        features_array.append( self.hog.extract_feature_window(window).astype(np.float64).ravel())
        features_array.append( self.sp.extract(p0).astype(np.float64).ravel())
        features_array.append( self.hist.extract(window).astype(np.float64).ravel())
        feature_vec = np.hstack(features_array).ravel()
        assert(np.isfinite(feature_vec).all())
        return feature_vec


class SpatialFeatureExtractor:

    def __init__(self, size): 
        self.size = size

    def preprocess(self, img, window_size):
        self.scale = (self.size/float(window_size))
        if self.scale==1:
            self.scaled=img
        else:
            self.scaled=cv2.resize( img, dsize=None, fx=self.scale, fy=self.scale )

    def extract(self, p0):
        p0 = (round(p0[0]*self.scale), round(p0[1]*self.scale))
        return extract_window( self.scaled, (p0, (p0[0]+self.size, p0[1]+self.size)))

class ColorHistFeatureExtractor:
    def __init__(self, cspace, nbins):
        self.cspace = cspace
        self.nbins = nbins

    def preprocess(self, img):
        self.cimg = convert_color(img, self.cspace)

    def extract(self, window):
        feature=[]
        for ch in range(self.cimg.shape[2]):
            hist, bin_edges = np.histogram( self.cimg[window[0][1]:window[1][1],window[0][0]:window[1][0]], self.nbins )
            feature.append(hist)
        return np.asarray(feature).ravel() * (1./(window[1][1]-window[0][1])*(window[1][0]-window[0][0]))


class HogFeatureExtractor:
    def __init__(self, cspace, orient, pix_per_cell, cell_per_block, hog_channel):
        self.cspace = cspace
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.hog_channel = hog_channel

    def preprocess(self, img):
        self.hog_features = extract_feature( img, self.cspace, self.orient, self.pix_per_cell, self.cell_per_block, self.hog_channel, feature_vec=False )
        return self.hog_features

    def extract_feature_window(self, window):
        x0,y0 = window[0]
        x1,y1 = window[1]
        x0 = int(x0/self.pix_per_cell )
        y0 = int(y0/self.pix_per_cell )
        x1 = int(x0 + (self.pix_per_cell - self.cell_per_block + 1))
        y1 = int(y0 + (self.pix_per_cell - self.cell_per_block + 1))
        #print(window, self.hog_features.shape )
        assert(x1 <= self.hog_features.shape[2] )
        assert(y1 <= self.hog_features.shape[1] )
        return self.hog_features[ :, y0:y1, x0:x1 ].ravel()
