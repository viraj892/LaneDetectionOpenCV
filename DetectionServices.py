import cv2
import numpy as np


def getLanePoints(img, slopes_intercepts):
    imshape = img.shape

    if None not in slopes_intercepts:
        neg_points = [0, np.int(slopes_intercepts[0][0] * 0 + slopes_intercepts[0][1]), np.int(imshape[1] * 0.45),
                      np.int(slopes_intercepts[0][0] * np.int(imshape[1] * 0.46) + slopes_intercepts[0][1])]
        pos_points = [np.int(imshape[1] * 0.55),
                      np.int(slopes_intercepts[1][0] * imshape[1] * 0.55 + slopes_intercepts[1][1]), imshape[1],
                      np.int(slopes_intercepts[1][0] * imshape[1] + slopes_intercepts[1][1])]
    else:
        return None

    return [neg_points, pos_points]


# just keep the observations with slopes with 1.5 std dev
def filterAveragedLine(obs):
    return np.array(abs(obs - np.mean(obs)) < 7.0 * np.std(obs))


# calc average of lines
def getAveragedLane(lines):
    neg = np.empty([1, 3])
    pos = np.empty([1, 3])

    # calculate slopes for each line to identify the positive and negative lines
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = float(float(y2 - y1) / float(x2 - x1))
            # a = y2 - y1
            # b = x2 - x1
            # print float(float(a)/float(b))
            intercept = y1 - slope * x1
            line_length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            if slope < 0:
                neg = np.append(neg, np.array([[slope, intercept, line_length]]), axis=0)
            elif slope > 0:
                pos = np.append(pos, np.array([[slope, intercept, line_length]]), axis=0)

    neg = neg[filterAveragedLine(neg[:, 0])]
    pos = pos[filterAveragedLine(pos[:, 0])]

    # weighted average of the slopes and intercepts based on the length of the line segment
    if len(neg[1:, 2]) > 0:
        neg_lines = np.dot(neg[1:, 2], neg[1:, :2]) / np.sum(neg[1:, 2])
    else:
        neg_lines = None

    if len(pos[1:, 2]) > 0:
        pos_lines = np.dot(pos[1:, 2], pos[1:, :2]) / np.sum(pos[1:, 2])
    else:
        pos_lines = None

    return neg_lines, pos_lines


def superImposeLaneOnFrame(img, endpoints, color=[0, 255, 0], thickness=7):
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    poly_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    print endpoints
    ## obtain slopes, intercepts, and endpoints of the weighted average line segments
    if endpoints is not None:
        for line in endpoints:
            ## draw lane lines
            cv2.line(line_img, (line[0], line[1]), (line[2], line[3]), color, thickness)
            # cv2.fillConvexPoly(poly_img, poly_points, color)

        for i in range(len(endpoints) - 1):
            point1 = [endpoints[0][0], endpoints[0][1]]
            point2 = [endpoints[0][2], endpoints[0][3]]
            point3 = [endpoints[1][0], endpoints[1][1]]
            point4 = [endpoints[1][2], endpoints[1][3]]

            poly_points = np.array([point1, point2, point3, point4, point1], np.int32)
            print "line1=", endpoints[0]
            print "line2=", endpoints[1]
            cv2.fillConvexPoly(line_img, poly_points, color)

    return line_img


def getPreviousMeanLine(lane_lines, prev_lane_lines):
    ## add the new lane line
    if lane_lines is not None:
        prev_lane_lines.append(lane_lines)

    ## only keep the 10 most recent lane lines
    if len(prev_lane_lines) >= 10:
        prev_lane_lines.pop(0)

    ## take the average of the past lane lines and the new ones
    if len(prev_lane_lines) > 0:
        return np.mean(prev_lane_lines, axis=0, dtype=np.int)
