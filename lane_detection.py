# importing some useful packages
import time

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
from moviepy.editor import VideoFileClip
from IPython.display import HTML

# %matplotlib inline

grayscale_img = None
gaussian_img = None
canny_img = None


def grayscale(img):
    """Applies the Grayscale transform
    This will return an input_image with only one color channel
    but NOTE: to see the returned input_image as grayscale
    (assuming your grayscaled input_image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an input_image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an input_image mask.

    Only keeps the region of the input_image defined by the polygon
    formed from `vertices`. The rest of the input_image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input input_image
    if len(img.shape) > 2:
        # i.e. 3 or 4 depending on your input_image
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the input_image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=8):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the input_image inplace (mutates the input_image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    # identifying positive and negative slopes of lines
    left_lines = []  # left lines are with positive slope
    right_lines = []  # right lines are with negative slope
    for line in lines:
        for x1, y1, x2, y2 in line:
            if float(y2 - y1) / float(x2 - x1) > 0:
                right_lines.append(line)
            else:
                left_lines.append(line)

    # find the lowest x and the highest y in the
    if left_lines:
        lowest_point_left = [left_lines[0][0][0], left_lines[0][0][1]]
        highest_point_left = [left_lines[0][0][2], left_lines[0][0][3]]
        for line in left_lines:
            for x1, y1, x2, y2 in line:
                if y1 >= lowest_point_left[1]:
                    lowest_point_left = [x1, y1]
                if y2 <= highest_point_left[1]:
                    highest_point_left = [x2, y2]

        cv2.line(img, (lowest_point_left[0], lowest_point_left[1]), (highest_point_left[0], highest_point_left[1]),
                 color,
                 thickness)

    if right_lines:
        highest_point_right = [right_lines[0][0][0], right_lines[0][0][1]]
        lowest_point_right = [right_lines[0][0][2], right_lines[0][0][3]]
        for line in right_lines:
            for x1, y1, x2, y2 in line:
                if y2 >= lowest_point_right[1] and x2 >= lowest_point_right[0]:
                    lowest_point_right = [x2, y2]
                if y1 <= highest_point_right[1]:
                    highest_point_right = [x1, y1]

        cv2.line(img, (lowest_point_right[0], lowest_point_right[1]), (highest_point_right[0], highest_point_right[1]),
                 color,
                 thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an input_image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    The result input_image is computed as follows:
    initial_img * α + img * β + γ
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


def lane_detection(input_image):
    kernel_size = 5

    low_threshold = 50
    high_threshold = 150

    rho = 2  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    # minimum number of votes (intersections in Hough grid cell)
    threshold = 15
    min_line_length = 40  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectible line segments

    global grayscale_img
    global canny_img
    global gaussian_img

    # grayscale input_image
    grayscale_img = grayscale(input_image)

    # gaussian blur
    gaussian_img = gaussian_blur(grayscale_img, kernel_size)

    # canny edge detection
    canny_img = canny(gaussian_img, low_threshold, high_threshold)

    # region masking
    imshape = input_image.shape

    img_width = float(imshape[1])
    img_height = float(imshape[0])

    # parametrize the corner parameters
    height_scale = 1.66

    roi_top_center = 480
    roi_offset = 30

    top_left_corner = (roi_top_center - roi_offset, img_height / height_scale)
    top_right_corner = (roi_top_center + roi_offset, img_height / height_scale)

    vertices = np.array([[(0, imshape[0]), top_left_corner, top_right_corner, (imshape[1], imshape[0])]],
                        dtype=np.int32)
    masked_edges = region_of_interest(canny_img, vertices)

    # hough transformation
    line_img = hough_lines(masked_edges, rho, theta,
                           threshold, min_line_length, max_line_gap)

    # drawing output
    output_img = weighted_img(input_image, line_img)
    return output_img


"""
#######################        TEST PART OF THE CODE        ##################################
"""
"""
# image = mpimg.imread('test_images/challenge.jpg')
# image = mpimg.imread('test_images/solidYellowCurve2.jpg')
image = mpimg.imread('test_images/solidWhiteRight.jpg')
# image = mpimg.imread('test_images/whiteCarLaneSwitch.jpg')
lane_detection_img = lane_detection(image)

# plt.imshow(lane_detection_img)
# plt.show()

# cv2.imshow('Grayscale', grayscale_img)
# cv2.imshow('Canny', canny_img)
# cv2.imshow('Gaussian', gaussian_img)

plt.imshow(lane_detection_img)
plt.show()

cv2.waitKey(0)  # waits until a key is pressed
cv2.destroyAllWindows()  # destroys the window showing image
"""

# """
# white_output = 'test_videos_output/solidYellowLeft.mp4'
# clip1 = VideoFileClip("test_videos/solidYellowLeft.mp4")

white_output = 'test_videos_output/solidWhiteRight.mp4'
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")

# white_output = 'test_videos_output/challenge.mp4'
# clip1 = VideoFileClip("test_videos/challenge.mp4")

# NOTE: this function expects color images!!
white_clip = clip1.fl_image(lane_detection)
white_clip.write_videofile(white_output, audio=False)
# """
