import cv2
import numpy as np


def region_of_interest(img):
    point1 = (round(img.shape[1] * 0.18), round(img.shape[0] * 0.90))
    point2 = (round(img.shape[1] * 0.43), round(img.shape[0] * 0.67))
    point3 = (round(img.shape[1] * 0.60), round(img.shape[0] * 0.67))
    point4 = (round(img.shape[1] * 0.95), round(img.shape[0] * 0.90))
    roi = np.array([point1, point2, point3, point4], dtype=np.int32)
    mask = np.zeros_like(img)
    mask_value = 100
    cv2.fillConvexPoly(mask, roi, mask_value)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


def enhanceLaneColor(image):
    HSL_img=image
    # convert image to HSL format (Hue, Saturation, Lightness
    # HSL_img = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    # cv2.imshow('hsl',HSL_img)

    # Keep only intensities falling in the white color range
    white_enhanced_img = cv2.inRange(HSL_img, np.uint8([135,135,135]),
                                     np.uint8([255, 255, 255]))
    # Keep only intensities falling in the yellow color range
    yellow_enhanced_img = cv2.inRange(HSL_img, np.uint8([10, 50, 100]), np.uint8([180, 255, 255]))

    # combine the yellow and white ranged images
    white_yellow = cv2.bitwise_or(white_enhanced_img, yellow_enhanced_img)

    # keep only enhanced intensities in the original image
    return cv2.bitwise_and(image, image, mask=white_yellow)


def get_weighted_image(img, initial_img, alpha=0.9, beta=0.95, gamma=0.):
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)


def applyGaussian(img, k, sigma=1):
    return cv2.GaussianBlur(img, (k, k), sigma)


def applyCanny(img, threshold1=50, threshold2=100, aperture=3):
    return cv2.Canny(img, threshold1, threshold2, aperture)
