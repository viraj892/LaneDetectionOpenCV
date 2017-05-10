import cv2
import DetectionServices as ds
import ImageProcessor as ip
import time
import numpy as np

filename = 'mp4/solidWhiteRight.mp4'
cap = cv2.VideoCapture(filename)
cached_lane_lines = []
count = 0

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    shape = frame.shape
    dashcam_flag = False
    if filename is 'mp4/dash_cam1.mp4':
        dashcam_flag = True
    # Our operations on the frame come here
    HSL_img = ip.enhanceLaneColor(frame, dashcam_flag)
    gaussian_img = ip.applyGaussian(HSL_img, k=3, sigma=1)
    canny_img = ip.applyCanny(gaussian_img, threshold1=0, threshold2=100, aperture=3)

    # apply region of interest
    roi_img = ip.region_of_interest(canny_img, flag=dashcam_flag)
    # cv2.imshow('roi', roi_img)
    lines = ds.applyHoughP(roi_img, rho=2, theta=np.pi / 180, threshold=130, min_line_length=30, max_line_gap=10)
    filtered_lines = ds.filterHoughLines(lines)

    final_lane_lines = ds.getAveragedLane(filtered_lines)
    final_lane_points = ds.getLanePoints(frame, final_lane_lines)
    lane_img = ds.superimpose_lane_on_frame(frame,
                                            endpoints=ds.getPreviousMeanLine(final_lane_points, cached_lane_lines))

    final_img = ip.get_weighted_image(lane_img, frame)
    cv2.imshow('lanes', final_img)

    # if ret:
    #     frame = cv2.flip(frame, 0)
    # out = cv2.VideoWriter('output.avi', -1, 25.0, (final_img.shape[0]*2, final_img.shape[1]*2), True)
    # out.write(final_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
