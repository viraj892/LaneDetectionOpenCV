import cv2
import numpy as np


def applyHoughP(img, rho=2, theta=np.pi / 180, threshold=170, min_line_length=50, max_line_gap=20):
    return cv2.HoughLinesP(img, rho, theta, threshold, min_line_length, max_line_gap)


def filterHoughLines(lines):
    alllines = []
    # nodetect = []
    if lines is not None:
        for line in lines:
            if line is not None:
                alllines.append(line)
    else:
        print "No lines detected"
        # nodetect.append(frame)

    # for i in range(len(nodetect)):
    #     count = count + 1
    #     name = "nodetect" + str(count) + ".jpg"
    #     print name
    #     cv2.imwrite(name, nodetect[i])
    # print alllines

    return alllines


def getLanePoints(img, lane_lines):
    imshape = img.shape

    if None not in lane_lines:
        neg_points = [0, np.int(lane_lines[0][0] * 0 + lane_lines[0][1]), np.int(imshape[1] * 0.45),
                      np.int(lane_lines[0][0] * np.int(imshape[1] * 0.46) + lane_lines[0][1])]
        pos_points = [np.int(imshape[1] * 0.55),
                      np.int(lane_lines[1][0] * imshape[1] * 0.55 + lane_lines[1][1]), imshape[1],
                      np.int(lane_lines[1][0] * imshape[1] + lane_lines[1][1])]
    else:
        return None

    return [neg_points, pos_points]


# just keep the observations with slopes with 1.5 std dev
def filterAveragedLine(slope):
    delta = abs(slope - np.mean(slope))
    return np.array(delta < 1.5 * np.std(slope))


# calc average of lines
def getAveragedLane(lines):
    neg = np.empty([1, 3])
    pos = np.empty([1, 3])

    # calculate slopes for each line to identify the positive and negative lines
    for line in lines:
        for x1, y1, x2, y2 in line:
            a = float(y2 - y1)
            b = float(x2 - x1)
            m = float(a / b)  # slope
            c = y1 - m * x1  # intercept
            line_len = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

            # identify the right and left lane lines based on slope value
            if m < 0:
                line_params = np.array([[m, c, line_len]])
                neg = np.append(neg, line_params, axis=0)
            elif m > 0:
                line_params = np.array([[m, c, line_len]])
                pos = np.append(pos, line_params, axis=0)

    # normalize the right and left lines
    neg = neg[filterAveragedLine(neg[:, 0])]
    pos = pos[filterAveragedLine(pos[:, 0])]

    # weighted average of the slopes and intercepts based on the length of the line segment
    if len(neg[1:, 2]) > 0:
        left_neg_line = np.dot(neg[1:, 2], neg[1:, :2]) / np.sum(neg[1:, 2])
    else:
        left_neg_line = None

    if len(pos[1:, 2]) > 0:
        right_pos_lines = np.dot(pos[1:, 2], pos[1:, :2]) / np.sum(pos[1:, 2])
    else:
        right_pos_lines = None

    return left_neg_line, right_pos_lines


def superimpose_lane_on_frame(img, endpoints, color=[51, 51, 255], thickness=7):
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # obtain slopes, intercepts, and endpoints of the weighted average line segments
    if endpoints is not None:
        for line in endpoints:
            # draw lane lines
            cv2.line(line_img, (line[0], line[1]), (line[2], line[3]), color, thickness)

        for i in range(len(endpoints) - 1):
            point1 = [endpoints[0][0], endpoints[0][1]]
            point2 = [endpoints[0][2], endpoints[0][3]]
            point3 = [endpoints[1][0], endpoints[1][1]]
            point4 = [endpoints[1][2], endpoints[1][3]]

            poly_points = np.array([point1, point2, point3, point4, point1], np.int32)
            # print "line1=", endpoints[0]
            # print "line2=", endpoints[1]
            cv2.fillConvexPoly(line_img, poly_points, color)

    return line_img


def getPreviousMeanLine(lane_lines, cached_lane_lines):
    # add the new lane line
    if lane_lines is not None:
        cached_lane_lines.append(lane_lines)

    # only keep the 10 most recent lane lines
    if len(cached_lane_lines) >= 10:
        cached_lane_lines.pop(0)

    # take the average of the past lane lines and the new ones
    if len(cached_lane_lines) > 0:
        return np.mean(cached_lane_lines, axis=0, dtype=np.int)
