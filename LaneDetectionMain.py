import cv2
import DetectionServices as ds
import ImageProcessor as ip
import time
import numpy as np

cap = cv2.VideoCapture('mp4/dash_cam1.mp4')
cached_lane_lines = []
count = 0
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    shape = frame.shape

    # Our operations on the frame come here
    HSL_img = ip.enhanceLaneColor(frame)
    cv2.imshow('hsl',HSL_img)
    gaussian_img = ip.applyGaussian(HSL_img, k=5, sigma=1)
    canny_img = ip.applyCanny(gaussian_img, threshold1=50, threshold2=100, aperture=3)

    # apply region of interest
    roi_img = ip.region_of_interest(canny_img)

    lines = ds.applyHoughP(roi_img, rho=2, theta=np.pi / 180, threshold=170, min_line_length=50, max_line_gap=20)
    filtered_lines = ds.filterHoughLines(lines)

    final_lane_lines = ds.getAveragedLane(filtered_lines)
    final_lane_points = ds.getLanePoints(frame, final_lane_lines)

    lane_img = ds.superimpose_lane_on_frame(frame,
                                            endpoints=ds.getPreviousMeanLine(final_lane_points, cached_lane_lines))

    final_img = ip.get_weighted_image(lane_img, frame)
    cv2.imshow('lanes', final_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
