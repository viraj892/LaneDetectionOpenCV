import cv2
import numpy as np


def region_of_interest(img, flag):
    if flag:
        point1 = (round(img.shape[1] * 0.18), round(img.shape[0] * 1))
        point2 = (round(img.shape[1] * 0.43), round(img.shape[0] * 0.68))
        point3 = (round(img.shape[1] * 0.50), round(img.shape[0] * 0.68))
        point4 = (round(img.shape[1] * 0.75), round(img.shape[0] * 1))
    else:
        point1 = (round(img.shape[1] * 0.18), round(img.shape[0] * 0.90))
        point2 = (round(img.shape[1] * 0.43), round(img.shape[0] * 0.67))
        point3 = (round(img.shape[1] * 0.60), round(img.shape[0] * 0.67))
        point4 = (round(img.shape[1] * 0.95), round(img.shape[0] * 0.90))
    roi = np.array([point1, point2, point3, point4], dtype=np.int32)
    mask = np.zeros_like(img)
    mask_value = 255
    cv2.fillConvexPoly(mask, roi, mask_value)
    masked_img = cv2.bitwise_and(img, mask)
    cv2.imshow('mask', mask)
    return masked_img


def enhanceLaneColor(image, flag):
    if flag:
        # color extraction values for dash_cam.mp4
        white_max = np.uint8([125, 35, 200])
        white_min = np.uint8([110, 20, 140])
        yellow_max = np.uint8([25, 90, 175])
        yellow_min = np.uint8([10, 25, 140])
    else:
        white_max = np.uint8([135, 15, 255])
        white_min = np.uint8([105, 0, 190])
        yellow_max = np.uint8([35, 165, 245])
        yellow_min = np.uint8([10, 80, 175])

    # convert image to HSV format (Hue, Saturation, Value)
    HSV_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Keep only intensities falling in the white color range
    white_enhanced_img = cv2.inRange(HSV_img, white_min, white_max)
    # cv2.imshow('white enhanced', white_enhanced_img)

    # Keep only intensities falling in the yellow color range
    yellow_enhanced_img = cv2.inRange(HSV_img, yellow_min, yellow_max)
    # cv2.imshow('yellow enhanced', yellow_enhanced_img)

    # combine the yellow and white ranged images
    white_yellow = cv2.bitwise_or(white_enhanced_img, yellow_enhanced_img)
    # cv2.imshow('wy', white_yellow)

    # keep only enhanced intensities in the original image
    return cv2.bitwise_and(image, image, mask=white_yellow)


def get_weighted_image(img, initial_img, alpha=0.9, beta=0.95, gamma=0.):
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)


def applyGaussian(img, k, sigma=1):
    return cv2.GaussianBlur(img, (k, k), sigma)


def applyCanny(img, threshold1=50, threshold2=100, aperture=3):
    return cv2.Canny(img, threshold1, threshold2, aperture)
