# Built in packages
import math
import sys
import os

# Matplotlib will need to be installed if it isn't already. This is the only package allowed for this base part of the 
# assignment.
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.image import imread

# import our basic, light-weight png reader library
import imageIO.png

# extras
import cv2
import numpy as np

# Define constant and global variables
TEST_MODE = False    # Please, DO NOT change this variable!

def readRGBImageToSeparatePixelArrays(input_filename):
    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)

# a useful shortcut method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):
    new_pixel_array = []
    for _ in range(image_height):
        new_row = []
        for _ in range(image_width):
            new_row.append(initValue)
        new_pixel_array.append(new_row)

    return new_pixel_array


###########################################
### You can add your own functions here ###
###########################################

def rgb_image_to_greyscale(image_width, image_height, px_array_r, px_array_g, px_array_b):
    px_array = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(image_height):
        for j in range(image_width):
            px_array[i][j] = round(0.3 * px_array_r[i][j]) + round(0.6 * px_array_g[i][j]) + round(0.1 * px_array_b[i][j])
    return px_array

def get_grayscale_histogram(image_width, image_height, px_array):
    histogram = [0] * 256
    for i in range(image_height):
        for j in range(image_width):
            value = round(px_array[i][j])
            if value < 0:
                value = 0
            elif value > 255:
                value = 255
            histogram[value] += 1
    return histogram

def get_cumulative_histogram(histogram):
    cumulative_histogram = [0] * 256
    cumulative_histogram[0] = histogram[0]
    for i in range(1, 256):
        cumulative_histogram[i] = cumulative_histogram[i - 1] + histogram[i]
    return cumulative_histogram

def percentile_based_contrast_stretching(image_width, image_height, px_array, alpha, beta):
    histogram = get_grayscale_histogram(image_width, image_height, px_array)
    cumulative_histogram = get_cumulative_histogram(histogram)
    image_size = image_width * image_height
    
    # find the smallest value qa such that C(qa) is larger than alpha% of the total number of pixels
    qa = 0
    for i in range(256):
        if cumulative_histogram[i] > alpha * image_size:
            qa = i
            break
    
    # find the largest value qb such that C(qb) is smaller than beta% of the total number of pixels
    qb = 255
    for i in range(255, -1, -1):
        if cumulative_histogram[i] < beta * image_size:
            qb = i
            break

    # apply the contrast stretching transformation
    for i in range(image_height):
        for j in range(image_width):
            mapping = 255 / (qb - qa) * (px_array[i][j] - qa)
            if mapping < 0:
                px_array[i][j] = 0
            elif mapping > 255:
                px_array[i][j] = 255
            else:
                px_array[i][j] = mapping

    return px_array

def laplacian_filter(image_width, image_height, px_array):
    kernel = [[1.0, 1.0, 1.0], [1.0, -8.0, 1.0], [1.0, 1.0, 1.0]]
    new_px_array = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(1, image_height - 1):
        for j in range(1, image_width - 1):
            sum = 0
            for k in range(3):
                for l in range(3):
                    sum += kernel[k][l] * px_array[i - 1 + k][j - 1 + l]
            new_px_array[i][j] = abs(sum)
    return new_px_array

def mean_filter(image_width, image_height, px_array):
    new_px_array = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(2, image_height - 2):
        for j in range(2, image_width - 2):
            sum = 0
            for k in range(5):
                for l in range(5):
                    sum += px_array[i - 2 + k][j - 2 + l]
            new_px_array[i][j] = abs(sum / 25)
    return new_px_array
    
def simple_thresholding(image_width, image_height, px_array, threshold):
    new_px_array = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(image_height):
        for j in range(image_width):
            new_px_array[i][j] = 255 if px_array[i][j] > threshold else 0
    return new_px_array

def adaptive_thresholding(image_width, image_height, px_array):
    histogram = get_grayscale_histogram(image_width, image_height, px_array)
    
    weighted_sum = 0
    for i in range(len(histogram)):
        weighted_sum += i * histogram[i]
    theta0 = weighted_sum / (image_width * image_height)

    while True:
        theta0 = round(theta0)
        sum_ob = 0
        sum_bg = 0
        weighted_sum_ob = 0
        weighted_sum_bg = 0
        for i in range(len(histogram)):
            if i < theta0:
                sum_ob += histogram[i]
                weighted_sum_ob += i * histogram[i]
            else:
                sum_bg += histogram[i]
                weighted_sum_bg += i * histogram[i]
        mean_ob = weighted_sum_ob / sum_ob
        mean_bg = weighted_sum_bg / sum_bg
        mean_mean = round((mean_ob + mean_bg) / 2)
        if theta0 == mean_mean:
            break
        theta0 = mean_mean

    return simple_thresholding(image_width, image_height, px_array, theta0)
    

def dilation(image_width, image_height, px_array):
    kernel = [[0, 0, 1, 0, 0], 
              [0, 1, 1, 1, 0], 
              [1, 1, 1, 1, 1], 
              [0, 1, 1, 1, 0], 
              [0, 0, 1, 0, 0]]
    
    new_px_array = createInitializedGreyscalePixelArray(image_width, image_height)
    
    for i in range(image_height):
        for j in range(image_width):
            new_px_array[i][j] = 0
            for k in range(5):
                for l in range(5):
                    # Calculate the coordinates of the pixel to access
                    ni = i + k - 2
                    nj = j + l - 2
                    if kernel[k][l] == 1:
                        if 0 <= ni < image_height and 0 <= nj < image_width:
                            # If the pixel is within bounds, check the pixel value
                            if px_array[ni][nj] == 255:
                                new_px_array[i][j] = 255
                                break
                        # No need to check for out-of-bounds pixels as they are treated as 0
                if new_px_array[i][j] == 255:
                    break
    
    return new_px_array


def erosion(image_width, image_height, px_array):
    kernel = [[0, 0, 1, 0, 0], 
              [0, 1, 1, 1, 0], 
              [1, 1, 1, 1, 1], 
              [0, 1, 1, 1, 0], 
              [0, 0, 1, 0, 0]]
    
    new_px_array = createInitializedGreyscalePixelArray(image_width, image_height)
    
    for i in range(image_height):
        for j in range(image_width):
            erosion_flag = True
            for k in range(5):
                for l in range(5):
                    # Calculate the coordinates of the pixel to access
                    ni = i + k - 2
                    nj = j + l - 2
                    if kernel[k][l] == 1:
                        if 0 <= ni < image_height and 0 <= nj < image_width:
                            # If the pixel is within bounds, check the pixel value
                            if px_array[ni][nj] == 0:
                                erosion_flag = False
                                break
                        else:
                            # If the pixel is out of bounds, treat it as 0
                            erosion_flag = False
                            break
                if not erosion_flag:
                    break
            new_px_array[i][j] = 255 if erosion_flag else 0
    
    return new_px_array

def queue_based_connected_component_labeling(image_width, image_height, px_array):
    label_array = createInitializedGreyscalePixelArray(image_width, image_height)
    label = 0
    queue = []
    for i in range(image_height):
        for j in range(image_width):
            if px_array[i][j] == 255 and label_array[i][j] == 0:
                label += 1
                queue.append((i, j))
                label_array[i][j] = label
                while len(queue) > 0:
                    (x, y) = queue.pop(0)
                    for k in range(-1, 2):
                        for l in range(-1, 2):
                            if x + k >= 0 and x + k < image_height and y + l >= 0 and y + l < image_width and px_array[x + k][y + l] == 255 and label_array[x + k][y + l] == 0:
                                queue.append((x + k, y + l))
                                label_array[x + k][y + l] = label
    return label_array

def get_bounding_boxes(image_width, image_height, label_array, num_labels):
    bounding_box_list = []
    for label in range(1, num_labels + 1):
        min_x = image_width
        min_y = image_height
        max_x = 0
        max_y = 0
        for i in range(image_height):
            for j in range(image_width):
                if label_array[i][j] == label:
                    min_x = min(min_x, j)
                    min_y = min(min_y, i)
                    max_x = max(max_x, j)
                    max_y = max(max_y, i)

        # check bounding box is roughly square
        top_length = max_x - min_x
        side_length = max_y - min_y
        threshold = 1.05
        if top_length > threshold * side_length or side_length > threshold * top_length:
            continue
        
        # check bounding box is not too small
        min_size = 100
        if top_length < min_size or side_length < min_size:
            continue
        
        bounding_box_list.append([min_x, min_y, max_x, max_y])
    return bounding_box_list

def get_bounding_box_area(bounding_box):
    return (bounding_box[2] - bounding_box[0]) * (bounding_box[3] - bounding_box[1])

def get_calibration_image_data():
    test_coins = os.listdir('Images/calibration')
    histograms = {}
    for coin in test_coins:
        current_coin = cv2.imread(f'Images/calibration/{coin}', cv2.IMREAD_COLOR)
        coin_hsv = cv2.cvtColor(current_coin, cv2.COLOR_BGR2HSV)
        coin_hist = cv2.calcHist([coin_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(coin_hist, coin_hist, 0, 255, cv2.NORM_MINMAX)
        histograms[coin] = coin_hist

    return histograms

def detect_coin_types(input_filename, bounding_box_list):
    image = cv2.imread(input_filename, cv2.IMREAD_COLOR)
    detected_coin_types = []

    calibration_histograms = get_calibration_image_data()
    
    # Calculate areas of all bounding boxes
    bounding_box_areas = [get_bounding_box_area(bbox) for bbox in bounding_box_list]
    
    for i, bounding_box in enumerate(bounding_box_list):
        metrics = []
        for coin in calibration_histograms:
            coin_hist = calibration_histograms[coin]

            coin_in_scene = image[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2]]
            coin_in_scene_hsv = cv2.cvtColor(coin_in_scene, cv2.COLOR_BGR2HSV)
            coin_in_scene_hist = cv2.calcHist([coin_in_scene_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
            cv2.normalize(coin_in_scene_hist, coin_in_scene_hist, 0, 255, cv2.NORM_MINMAX)

            metric_val = cv2.compareHist(coin_hist, coin_in_scene_hist, cv2.HISTCMP_CORREL)
            metrics.append([coin.split("_")[0], metric_val])

        metrics.sort(key=lambda x: x[1], reverse=True)
        weighted_metrics = {
            "10": 0,
            "20": 0,
            "50": 0,
            "1": 0,
            "2": 0
        }
        for idx, metric in enumerate(metrics):
          weighted_metrics[metric[0]] += (idx + 1) * metric[1]
        
        sorted_metrics = sorted(weighted_metrics.items(), key=lambda x: x[1], reverse=True)
        
        max_coin = metrics[0][0]
        coin_type = max_coin

        bbox_area = bounding_box_areas[i]


        # If the detected coin type is ambiguous, use the relative size to distinguish
        if max_coin in ["1", "2"]:
            similar_coins = [bounding_box_areas[j] for j in range(len(detected_coin_types)) if detected_coin_types[j] in ["1", "2"]]
            if not similar_coins:
                coin_type = max_coin
            elif bbox_area > max(similar_coins) * 1.1:
                coin_type = "2"
            else:
                coin_type = "1"
        elif max_coin in ["20", "50"]:
            similar_coins = [bounding_box_areas[j] for j in range(len(detected_coin_types)) if detected_coin_types[j] in ["20", "50"]]
            if not similar_coins:
                coin_type = max_coin
            elif bbox_area > max(similar_coins) * 1.1:
                coin_type = "50"
            else:
                coin_type = "20"

        detected_coin_types.append(coin_type)

    return detected_coin_types

def otsu_threshold(image_width, image_height, px_array):
    histogram = get_grayscale_histogram(image_width, image_height, px_array)

    total_pixels = image_width * image_height
    sum_all = sum(i * histogram[i] for i in range(256))
    
    sum_bg = 0
    weight_bg = 0
    weight_obj = 0
    max_variance = 0
    threshold = 0
    
    for i in range(256):
        weight_bg += histogram[i]
        if weight_bg == 0:
            continue
        weight_obj = total_pixels - weight_bg
        if weight_obj == 0:
            break
        
        sum_bg += i * histogram[i]
        mean_b = sum_bg / weight_bg
        mean_f = (sum_all - sum_bg) / weight_obj
        
        variance_between = weight_bg * weight_obj * ((mean_b - mean_f) ** 2)
        
        if variance_between > max_variance:
            max_variance = variance_between
            threshold = i
    
    return threshold

# This is our code skeleton that performs the coin detection.
def main(input_path, output_path):
    # This is the default input image, you may change the 'image_name' variable to test other images.
    image_name = 'hard_case_2'
    input_filename = f'./Images/hard/{image_name}.png'
    if TEST_MODE:
        input_filename = input_path

    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)
    
    ###################################
    ### STUDENT IMPLEMENTATION Here ###
    ###################################

    # Convert to greyscale
    greyscale_px_array = rgb_image_to_greyscale(image_width, image_height, px_array_r, px_array_g, px_array_b)
    
    # Perform percentile-based contrast stretching
    contrast_stretched_px_array = percentile_based_contrast_stretching(image_width, image_height, greyscale_px_array, 0.05, 0.95)
    
    # Perform Laplacian filter
    laplacian_px_array = laplacian_filter(image_width, image_height, contrast_stretched_px_array)
    
    # Perform mean filter 3 times
    mean_filtered_px_array = mean_filter(image_width, image_height, laplacian_px_array)
    mean_filtered_px_array = mean_filter(image_width, image_height, mean_filtered_px_array)
    mean_filtered_px_array = mean_filter(image_width, image_height, mean_filtered_px_array)

    otsu_threshold_value = otsu_threshold(image_width, image_height, mean_filtered_px_array)

    # Perform adaptive thresholding using histogram method
    thresholded_px_array = simple_thresholding(image_width, image_height, mean_filtered_px_array, otsu_threshold_value)

    # Perform dilation 6 times
    dilated_px_array = dilation(image_width, image_height, thresholded_px_array)
    dilated_px_array = dilation(image_width, image_height, dilated_px_array)
    dilated_px_array = dilation(image_width, image_height, dilated_px_array)
    dilated_px_array = dilation(image_width, image_height, dilated_px_array)
    dilated_px_array = dilation(image_width, image_height, dilated_px_array)
    dilated_px_array = dilation(image_width, image_height, dilated_px_array)
    
    # Perform erosion 6 times
    eroded_px_array = erosion(image_width, image_height, dilated_px_array)
    eroded_px_array = erosion(image_width, image_height, eroded_px_array)
    eroded_px_array = erosion(image_width, image_height, eroded_px_array)
    eroded_px_array = erosion(image_width, image_height, eroded_px_array)
    eroded_px_array = erosion(image_width, image_height, eroded_px_array)
    eroded_px_array = erosion(image_width, image_height, eroded_px_array)

    # Perform connected component labeling
    label_array = queue_based_connected_component_labeling(image_width, image_height, eroded_px_array)
    num_labels = max([max(row) for row in label_array])

    # Get bounding box list
    bounding_box_list = get_bounding_boxes(image_width, image_height, label_array, num_labels)
    print(f'Number of coins: {len(bounding_box_list)}')

    # sort bounding boxes by area
    bounding_box_list.sort(key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)

    # Detect coin types
    coin_type_list = detect_coin_types(input_filename, bounding_box_list)
        
    ############################################
    ### Bounding box coordinates information ###
    ### bounding_box[0] = min x
    ### bounding_box[1] = min y
    ### bounding_box[2] = max x
    ### bounding_box[3] = max y
    ############################################

    fig, axs = pyplot.subplots(1, 1)
    
    # Loop through all bounding boxes
    coin_type_index = 0
    for bounding_box in bounding_box_list:
        bbox_min_x = bounding_box[0]
        bbox_min_y = bounding_box[1]
        bbox_max_x = bounding_box[2]
        bbox_max_y = bounding_box[3]
        
        bbox_xy = (bbox_min_x, bbox_min_y)
        bbox_width = bbox_max_x - bbox_min_x
        bbox_height = bbox_max_y - bbox_min_y
        rect = Rectangle(bbox_xy, bbox_width, bbox_height, linewidth=2, edgecolor='r', facecolor='none')
        axs.add_patch(rect)

        coin_type = coin_type_list[coin_type_index]
        axs.text(bbox_min_x, bbox_min_y - 15, ("$" if coin_type in ["1", "2"] else "") + coin_type_list[coin_type_index] + ("¢" if coin_type in ["10", "20", "50"] else ""), fontsize=12, color='b')
        coin_type_index += 1

    pyplot.axis('off')
    pyplot.tight_layout()
    default_output_path = f'./output_images/{image_name}_with_bbox.png'
    if not TEST_MODE:
        # Saving output image to the above directory
        image = imread(input_filename)
        pyplot.imshow(image, aspect='equal')
        # pyplot.imshow(dilated_px_array, aspect='equal', cmap="gray")
        pyplot.savefig(default_output_path, bbox_inches='tight', pad_inches=0)

        # Show image with bounding box on the screen
        pyplot.show()
    else:
        # Please, DO NOT change this code block!
        pyplot.savefig(output_path, bbox_inches='tight', pad_inches=0)



if __name__ == "__main__":
    num_of_args = len(sys.argv) - 1
    
    input_path = None
    output_path = None
    if num_of_args > 0:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        TEST_MODE = True
    
    main(input_path, output_path)
    