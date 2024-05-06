
import cv2
import numpy as np
from skimage import color
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split


def extract_sift_features(training_set):
    # Extract SIFT features from the training set
    descriptor = cv2.SIFT_create()
    # extract keypoints from each image
    training_set['keypoints'] = []
    training_set['descriptors'] = []

    for img in training_set['data']:
        gray = np.uint8(color.rgb2gray(img)*255)
        kp, des = descriptor.detectAndCompute(gray, None)
        training_set['keypoints'].append(kp)
        training_set['descriptors'].append(des)
    return training_set
        

def extract_bag_of_sift(training_set, k = 50, batch_size = 100):

        # initiate a MiniBatchKMeans object
        kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size)

        # concatenate all descriptors into a single array
        all_descriptors = np.concatenate(training_set['descriptors'])

        # fit the kmeans model
        kmeans.fit(all_descriptors)
        training_set['histograms'] = []


        for des in training_set['descriptors']:
            # predict the cluster index for each descriptor
            clusters = kmeans.predict(des)
            # compute the histogram
            hist, _ = np.histogram(clusters, bins=np.arange(k+1))

            training_set['histograms'].append(hist)
    

        return training_set


def extract_color_histograms(training_set):
     training_set['color_histograms'] = []

     for img in training_set['data']:
        # convert the image to HSV
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = color.rgb2hsv(img)
        # compute the histogram
        hist,_ = np.histogram(img[:,:,0].ravel(), bins=np.arange(0, 1.01, 0.02))
        hist = hist/hist.sum()
        training_set['color_histograms'].append(hist)
     return training_set

def extract_lbp_histograms(training_set):
     pass


def split_dataset(training_set):
    X =  np.concatenate([training_set['histograms'], training_set['color_histograms']], axis=1)
    y = training_set['label']
    # split the data into training and testing sets

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


