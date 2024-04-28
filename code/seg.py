# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
import skimage.measure as measure

def segmentation(image_path,save_index,optimal_s = 801, optimal_t = -11,filter_index = 300):
    image  = cv2.imread(image_path)  # read image
    # resize image
    image = cv2.resize(image, (0, 0), fx=0.1, fy=0.1)

    # # apply gaussian
    image = cv2.GaussianBlur(image, (5, 5), 0)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    # invert the image
    gray_image = cv2.bitwise_not(gray_image)
    # convert the image to float
    gray_image = gray_image.astype(float)


    local = filters.threshold_local(gray_image, optimal_s, offset=optimal_t)
    bina_local  = gray_image > local

    contours = measure.find_contours(bina_local, 0.8)


    # now i want to find the area of the contours
    areas = []
    for contour in contours:
        areas.append(len(contour))


    filtered_contours = [contour for contour in contours if len(contour) > filter_index]
    # now cut off the region of each contour
    cut_images = []
    for filtered_contour in filtered_contours:
        min_x, max_x = int(np.min(filtered_contour[:, 1])), int(np.max(filtered_contour[:, 1]))
        min_y, max_y = int(np.min(filtered_contour[:, 0])), int(np.max(filtered_contour[:, 0]))
        cut_image = image[min_y:max_y, min_x:max_x]
        cut_images.append(cut_image)

    # store the cutted images
    for i, cut_image in enumerate(cut_images):
        cv2.imwrite(f"images/cutted_dishes/cutted_dish_{save_index}_{i}.jpg",cut_image)