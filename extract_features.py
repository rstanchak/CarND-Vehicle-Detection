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

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_feature(rgb, cspace='RGB', orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0, feature_vec=True):
    # apply color conversion if other than 'RGB'
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(rgb, cv2.COLOR_BGR2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(rgb, cv2.COLOR_BGR2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(rgb, cv2.COLOR_BGR2YUV)
        elif cspace == 'YCrCb':
            feature_image = cv2.cvtColor(rgb, cv2.COLOR_BGR2YCrCb)
    else: feature_image = np.copy(rgb)

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
