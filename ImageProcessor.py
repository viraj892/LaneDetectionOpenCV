import cv2
import numpy as np


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillConvexPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    cv2.imshow('roi', masked_image)
    return masked_image


def color_selection(image):
    hls_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    white_color = cv2.inRange(hls_image, np.uint8([20, 200, 0]),
                              np.uint8([255, 255, 255]))  ## note that OpenCV uses BGR not RGB
    yellow_color = cv2.inRange(hls_image, np.uint8([10, 50, 100]), np.uint8([100, 255, 255]))

    combined_color_images = cv2.bitwise_or(white_color, yellow_color)
    return cv2.bitwise_and(image, image, mask=combined_color_images)


def weighted_img(img, initial_img, alpha=0.9, beta=0.95, gamma=0.):
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)

def applyGaussian(img, k, sigma=1):
    gaussian_blur = cv2.GaussianBlur(img, (k, k), sigma)
    return gaussian_blur
