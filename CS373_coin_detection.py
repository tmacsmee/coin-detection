# Built in packages
import math
import sys

# Matplotlib will need to be installed if it isn't already. This is the only package allowed for this base part of the 
# assignment.
from matplotlib import pyplot
from matplotlib.patches import Rectangle

# import our basic, light-weight png reader library
import imageIO.png

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
            histogram[px_array[i][j]] += 1
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

def horizontal_scharr_filter(image_width, image_height, px_array):
    scharr_filter = [[3, 0, -3], [10, 0, -10], [3, 0, -3]]
    new_px_array = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(1, image_height - 1):
        for j in range(1, image_width - 1):
            new_px_array[i][j] = 0
            for k in range(3):
                for l in range(3):
                    new_px_array[i][j] += px_array[i - 1 + k][j - 1 + l] * scharr_filter[k][l]
            new_px_array[i][j] = new_px_array[i][j] / 32
    return new_px_array

def vertical_scharr_filter(image_width, image_height, px_array):
    scharr_filter = [[3, 10, 3], [0, 0, 0], [-3, -10, -3]]
    new_px_array = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(1, image_height - 1):
        for j in range(1, image_width - 1):
            new_px_array[i][j] = 0
            for k in range(3):
                for l in range(3):
                    new_px_array[i][j] += px_array[i - 1 + k][j - 1 + l] * scharr_filter[k][l]
            new_px_array[i][j] = new_px_array[i][j] / 32
    return new_px_array

def combined_scharr_filter(image_width, image_height, horizontal_scharr_px_array, vertical_scharr_px_array):
    new_px_array = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(1, image_height - 1):
        for j in range(1, image_width - 1):
            new_px_array[i][j] = abs(vertical_scharr_px_array[i][j]) + abs(horizontal_scharr_px_array[i][j])
    return new_px_array

# 5x5 mean filter
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
    

# This is our code skeleton that performs the coin detection.
def main(input_path, output_path):
    # This is the default input image, you may change the 'image_name' variable to test other images.
    image_name = 'easy_case_1'
    input_filename = f'./Images/easy/{image_name}.png'
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

    # Perform horizontal Scharr filter
    horizontal_scharr_px_array = horizontal_scharr_filter(image_width, image_height, contrast_stretched_px_array)

    # Perform vertical Scharr filter
    vertical_scharr_px_array = vertical_scharr_filter(image_width, image_height, contrast_stretched_px_array)

    # Combine horizontal and vertical Scharr filters
    combined_scharr_px_array = combined_scharr_filter(image_width, image_height, horizontal_scharr_px_array, vertical_scharr_px_array)
    
    # Perform mean filter 3 times n loop
    mean_filtered_px_array = mean_filter(image_width, image_height, combined_scharr_px_array)
    mean_filtered_px_array = mean_filter(image_width, image_height, mean_filtered_px_array)
    mean_filtered_px_array = mean_filter(image_width, image_height, mean_filtered_px_array)
    px_array = mean_filtered_px_array
    
    ############################################
    ### Bounding box coordinates information ###
    ### bounding_box[0] = min x
    ### bounding_box[1] = min y
    ### bounding_box[2] = max x
    ### bounding_box[3] = max y
    ############################################
    
    bounding_box_list = [[150, 140, 200, 190]]  # This is a dummy bounding box list, please comment it out when testing your own code.
    
    fig, axs = pyplot.subplots(1, 1)
    axs.imshow(px_array, aspect='equal')
    
    # Loop through all bounding boxes
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
        
    pyplot.axis('off')
    pyplot.tight_layout()
    default_output_path = f'./output_images/{image_name}_with_bbox.png'
    if not TEST_MODE:
        # Saving output image to the above directory
        pyplot.savefig(default_output_path, bbox_inches='tight', pad_inches=0)
        
        # Show image with bounding box on the screen
        pyplot.imshow(px_array, cmap='gray', aspect='equal')
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
    