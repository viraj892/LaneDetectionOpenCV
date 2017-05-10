import cv2
import numpy as np
from time import sleep

def angle(x0, y0, x1, y1):
    delta_y = y1 - y0
    delta_x = x1 - x0
    return np.arctan2(delta_y, delta_x)

def vanishing_point(lines, max_len = 0):
    m0 = y_int0 = 0

    v_pts = []

    for line in lines:
        x0, y0, x1, y1 = line[0]

        delta_y = y1 - y0
        delta_x = x1 - x0

        theta = np.arctan2(delta_y, delta_x)
        a = np.cos(theta)
        b = np.sin(theta)

        x_tmp = a * 2
        y_tmp = b * 2

        x0 = x_tmp + 830 * -b
        y0 = y_tmp + 360 * a
        x1 = x_tmp - 830 * -b
        y1 = y_tmp - 360 * a

        m1 = delta_y/delta_x

        if (np.abs(m0 - m1) > 0):
            y_int1 = y1 - (m1 * x1)

            x_comp = np.abs((y_int1 - y_int0) / (m0 - m1))
            y_comp = np.abs(m1 * x_comp + y_int1)

            y_int0 = y_int1
            m0 = m1

            v_pts.append((int(x_comp), int(y_comp)))

    return v_pts


cap = cv2.VideoCapture('mp4/MAH00125.mp4')
filter_angle = 13

while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    image = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    sigma = 1
    gaussian_blur = cv2.GaussianBlur(gray, (5, 5), sigma)

    aperature = 3
    canny = cv2.Canny(gaussian_blur, 0, 255, aperature)

    rho = 2
    theta = np.pi / 180
    threshold = 170
    min_line_length = 50
    max_line_gap = 5

    lines = cv2.HoughLinesP(canny, rho, theta, threshold, min_line_length, max_line_gap)

    if lines != None:

        filtered = []

        for line in lines:
            x0, y0, x1, y1 = line[0]

            # filter out horizontal lines
            line_angle = angle(x0, y0, x1, y1) * (180/np.pi)
            if np.abs(line_angle) > filter_angle:
                # print line_angle
                cv2.line(image, (x0, y0), (x1, y1), (0, 255, 0), 3, cv2.LINE_AA)
                filtered.append(line)

        v_pts = vanishing_point(filtered)

        print v_pts
        print image.shape

        for pts in v_pts:
            cv2.circle(image, pts, 5, (0, 0, 255), 2)

    else:
        print "No lines"

    # Display the resulting frame
    cv2.imshow('frame', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()