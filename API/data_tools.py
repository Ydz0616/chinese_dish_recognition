import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
import skimage.measure as measure
from skimage.io import imread, imsave  
from cv2 import morphologyEx
import os
import random


# find contours, return contour number, and for each contour, return a tuple of its area and its perimeter ( returned in list), and the contours
def find_contours(image_path,kernel_size = (90,90)):
    image = cv2.imread(image_path)  # read image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    # invert the image
    gray_image = cv2.bitwise_not(gray_image)
    # convert the image to float
    gray_image = gray_image.astype(float)
    # apply gaussian filter
    gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    # apply otsu filter
    threshold = filters.threshold_otsu(gray_image)
    bina_otsu = gray_image > threshold

    # apply closing operation
    kernel = np.ones(kernel_size, np.uint8)
    closing = morphologyEx(bina_otsu.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    # opening kernel is set to be 15 to remove unwanted noise, together with the gaussian
    kernel = np.ones((15, 15), np.uint8)
    opening = morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    # detect contours
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #  convert image and store the contours
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    # return total number of contours, and for each contour, return a tuple of its area and its perimeter
    return len(contours), [(cv2.contourArea(contour), cv2.arcLength(contour,True),contour) for contour in contours],contours



def segment_images(image_path,save_path,save_index,kernel_size = (90,90),length_thres_low =300000, length_thres_high = 900000, area_thres_low = 2000, area_thres_high = 5000):
    num, info, contours = find_contours(image_path,kernel_size)
    image = cv2.imread(image_path)
    # print(info)
    # filter out the unwanted contours in info based on the length and area threshold, the info has the shape [(length,area),...   ]
    filtered_info = []
    for i in range(len(info)):
        if info[i][0] > length_thres_low and info[i][0] < length_thres_high and info[i][1] > area_thres_low and info[i][1] < area_thres_high:
            filtered_info.append(info[i])

    # print('the filterd info is',filtered_info)

    contours = [filtered_info[i][2] for i in range(len(filtered_info))]
    
    os.makedirs(save_path, exist_ok=True)
    cropped_images = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # crop the image
        cropped = image[y:y+h, x:x+w]
        cropped_images.append(cropped)

    
    for i, cut_image in enumerate(cropped_images):
        cv2.imwrite(f"{save_path}/segmented_dish_{save_index}_{i}.jpg",cut_image)


def load_dataset():
    # make training set a dictionary of data, label, numerical_label


    training_set = {'data':[], 'label':[]}



    # load all images from /Users/taojing/Desktop/image_classification/images/categorized_dishes/tomato_egg
    for file in os.listdir('images/training_set/tomato_egg'):
        if file.endswith('.jpg'):
            # read the image as data, add label
            training_set['data'].append(imread('images/training_set/tomato_egg/'+file))
            training_set['label'].append('tomato_egg')
            training_set['numerical_label'] = 0

    for file in os.listdir('images/training_set/hong_shao_rou'):
        if file.endswith('.jpg'):
            training_set['data'].append(imread('images/training_set/hong_shao_rou/'+file))
            training_set['label'].append('braised_pork')
            training_set['numerical_label'] = 1

    for file in os.listdir('images/training_set/purple_rice'):
        if file.endswith('.jpg'):
            training_set['data'].append(imread('images/training_set/purple_rice/'+file))
            training_set['label'].append('purple_rice')
            training_set['numerical_label'] = 2

    for file in os.listdir('images/training_set/shanghai_green'):
        if file.endswith('.jpg'):
            training_set['data'].append(imread('images/training_set/shanghai_green/'+file))
            training_set['label'].append('shanghai_greens')
            training_set['numerical_label'] = 3


    for file in os.listdir('images/training_set/shirmp_egg'):
        if file.endswith('.jpg'):
            training_set['data'].append(imread('images/training_set/shirmp_egg/'+file))
            training_set['label'].append('shrimp_ball')
            training_set['numerical_label'] = 4

    # Shuffle the data and labels simultaneously
    combined = list(zip(training_set['data'], training_set['label']))
    random.shuffle(combined)

    # Unzip the shuffled data and labels
    shuffled_data, shuffled_labels = zip(*combined)

    # Update the training_set dictionary with shuffled data and labels
    training_set['data'] = list(shuffled_data)
    training_set['label'] = list(shuffled_labels)
    training_set['numerical_label'] = list(shuffled_labels)

    return training_set
