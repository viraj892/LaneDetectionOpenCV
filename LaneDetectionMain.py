import cv2
import DetectionServices as ds
import ImageProcessor as ip
import time
import numpy as np

cap = cv2.VideoCapture('mp4/solidWhiteRight.mp4')
prev_lane_lines = []

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    # frame=frame[300:900,:]
    shape = frame.shape

    # Our operations on the frame come here
    image = ip.color_selection(frame)
    gaussian_blur = ip.applyGaussian(image, k=5, sigma=1)
    aperature = 3
    canny = cv2.Canny(gaussian_blur, 50, 100, aperature)

    # apply region of interest
    vertices = np.array(
        [(shape[1] * 0.1, shape[0]), (shape[1] * 0.45, shape[0] * 0.6), (shape[1] * .55, shape[0] * 0.6),
         (shape[1], shape[0])], dtype=np.int32)
    roi_img = ip.region_of_interest(canny, vertices=vertices)

    rho = 2
    theta = np.pi / 180
    threshold = 170
    min_line_length = 50
    max_line_gap = 20

    lines = cv2.HoughLinesP(roi_img, rho, theta, threshold, min_line_length, max_line_gap)
    # print lines
    alllines = []
    if lines is not None:
        for line in lines:
            if line is not None:
                alllines.append(line)
    else:
        print "No lines detected"
    # print alllines
    slopes_intercepts = ds.getAveragedLane(alllines)
    # print slopes_intercepts
    endpoints = ds.getLanePoints(frame, slopes_intercepts)

    lane_img = ds.superImposeLaneOnFrame(frame, endpoints=ds.getPreviousMeanLine(endpoints, prev_lane_lines))

    final_img = ip.weighted_img(lane_img, frame)
    # print lane_lines
    # cv2.imshow('lane_lines', lane_lines)
    cv2.imshow('lanes', final_img)
    cv2.imshow('orig', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
