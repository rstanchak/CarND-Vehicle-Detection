# file: vehicle_detection2.py
# author: Roman Stanchak (rstanchak@gmail.com)
# description: class to detect vehicles in an image using a feature extractor
#              and classifier
import numpy as np
import scipy.ndimage.measurements as spmeas
import cv2


class Shape:
    """fake class to pass to slide_window instead of actual nparray"""
    def __init__(self, shape):
        self.shape = shape


class VehicleDetector:
    """vehicle detection pipeline"""
    def __init__(self, classifier, feature_extractor):
        self.classifier = classifier
        self.feature_extractor = feature_extractor
        self.heatmap = None

    def draw_windows(self, output, windows, color, thickness):
        for w in windows:
            cv2.rectangle(output, w[0], w[1], color, thickness)

    def get_windows(self, img):
        """
        generate a list of lists of search windows for the given image at
        multiple scales
        """
        shape = Shape(img.shape)
        windows = []

        # 1:1 scale, search area that is further away, 50% overlap
        y0 = int(5*img.shape[0]/9)
        y1 = int(3*img.shape[0]/4)
        x0 = int(img.shape[1]/5)
        x1 = int(4*img.shape[1]/5)

        windows.append(slide_window(shape,
                                    x_start_stop=(x0, x1),
                                    y_start_stop=(y0, y1),
                                    xy_overlap=(0.5, 0.5)))

        # 1:2 scale, search intermediate area, full width, more overlap
        shape = Shape((int(shape.shape[0]/2), int(shape.shape[1]/2)))
        y0 = int(shape.shape[0]/2)
        y1 = int(3*shape.shape[0]/4)
        x0 = 0
        x1 = shape.shape[1]

        windows.append(slide_window(shape, x_start_stop=(x0, x1),
                                    y_start_stop=(y0, y1),
                                    xy_overlap=(0.75, 0.75)))

        # 1:4 scale, search width and to bottom of image
        shape = Shape((int(shape.shape[0]/2), int(shape.shape[1]/2)))
        y0 = int(shape.shape[0]/2)
        y1 = shape.shape[0]
        x0 = 0
        x1 = shape.shape[1]
        windows.append(slide_window(shape, x_start_stop=(x0, x1),
                                    y_start_stop=(y0, y1),
                                    xy_overlap=(0.9, 0.9)))

        return windows

    def process_image(self, img):
        """
        process the image by detecting vehicles and drawing a bounding box
        around each
        """
        output = np.copy(img)
        if self.heatmap is None:
            self.heatmap = np.zeros(img.shape[:2], dtype=np.float64)
        else:
            self.heatmap = np.multiply(self.heatmap, 0.75)

        windows = self.get_windows(img)
        self.detections = []

        scaled = img
        for logscale in range(0, 2):
            scale = (1 << logscale)
            if logscale > 0:
                scaled = cv2.pyrDown(scaled)

            self.feature_extractor.preprocess(scaled)
            for w in windows[logscale]:
                # Extract the test window from original image
                test_features = \
                        self.feature_extractor.extract_feature_window(w)
                test_features = test_features.reshape(1, -1)

                # Predict using classifier
                prediction = self.classifier.predict(test_features)
                # If positive (prediction == 1) then save the window
                p1 = (w[0][0]*scale, w[0][1]*scale)
                p2 = (w[1][0]*scale, w[1][1]*scale)
                if prediction == 1:
                    self.detections.append((p1, p2))
                else:
                    # cv2.rectangle(output, p1, p2, (255,0,0), 1)
                    pass

        # self.draw_windows( output, self.detections, (0,255,0), 2)
        self.heatmap = add_heat(self.heatmap, self.detections)
        mask = apply_threshold(np.copy(self.heatmap), 1./128)
        # return draw_heat(output, mask)
        labels = spmeas.label(mask)
        return draw_labeled_bboxes(output, labels)


def draw_heat(img, heatmap):
    """overlay the heatmap on the image"""
    mask = np.zeros(img.shape, dtype=np.uint8)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(heatmap)
    scale = 1.0
    if maxVal > 0:
        scale *= (255./maxVal)
    mask[:, :, 1] = np.multiply(heatmap, scale).astype(np.uint8)
    return cv2.addWeighted(img, 1.0, mask, 0.8, gamma=0.0)


def add_heat(heatmap, bbox_list):
    """add to the heatmap proportional to the size of the detection box"""
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += \
                (1./(box[1][0]-box[0][0]))

    # Return updated heatmap
    return heatmap


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap < threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)),
                (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 255, 0), 2)
    # Return the image
    return img


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    """
    take an image, start and stop positions in both x and y,
    window size (x and y dimensions), and overlap fraction (for both x and y)
    return a list of search windows
    """
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            if endx > img.shape[1]:
                # print(((startx, starty), (endx, endy)))
                continue
            if endy > img.shape[0]:
                # print(((startx, starty), (endx, endy)))
                continue
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list
