import cv2
import numpy as np


def gen_endpoints(img, slopes_intercepts):
    imshape = img.shape

    if None not in slopes_intercepts:
        neg_points = [0, np.int(slopes_intercepts[0][0] * 0 + slopes_intercepts[0][1]), np.int(imshape[1] * 0.45),
                      np.int(slopes_intercepts[0][0] * np.int(imshape[1] * 0.45) + slopes_intercepts[0][1])]
        pos_points = [np.int(imshape[1] * 0.55),
                      np.int(slopes_intercepts[1][0] * imshape[1] * 0.55 + slopes_intercepts[1][1]), imshape[1],
                      np.int(slopes_intercepts[1][0] * imshape[1] + slopes_intercepts[1][1])]
    else:
        return None

    return [neg_points, pos_points]


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


def to_keep_index(obs, std=1.5):
    return np.array(abs(obs - np.mean(obs)) < std * np.std(obs))


# calc average of lines
def avg_lines(lines):
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

    ## just keep the observations with slopes with 1.5 std dev
    neg = neg[to_keep_index(neg[:, 0])]
    pos = pos[to_keep_index(pos[:, 0])]

    ## weighted average of the slopes and intercepts based on the length of the line segment
    if len(neg[1:, 2]) > 0:
        neg_lines = np.dot(neg[1:, 2], neg[1:, :2]) / np.sum(neg[1:, 2])
    else:
        neg_lines = None

    if len(pos[1:, 2]) > 0:
        pos_lines = np.dot(pos[1:, 2], pos[1:, :2]) / np.sum(pos[1:, 2])
    else:
        pos_lines = None

    return neg_lines, pos_lines


def gen_lane_lines(img, endpoints, color=[0, 255, 0], thickness=7):
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    ## obtain slopes, intercepts, and endpoints of the weighted average line segments
    if endpoints is not None:
        for line in endpoints:
            ## draw lane lines
            cv2.line(line_img, (line[0], line[1]), (line[2], line[3]), color, thickness)

    return line_img


def find_mean_lines(lane_lines, prev_lane_lines):
    ## add the new lane line
    if lane_lines is not None:
        prev_lane_lines.append(lane_lines)

    ## only keep the 10 most recent lane lines
    if len(prev_lane_lines) >= 10:
        prev_lane_lines.pop(0)

    ## take the average of the past lane lines and the new ones
    if len(prev_lane_lines) > 0:
        return np.mean(prev_lane_lines, axis=0, dtype=np.int)


def weighted_img(img, initial_img, alpha=0.9, beta=0.95, gamma=0.):
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)


def color_selection(image):
    hls_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    white_color = cv2.inRange(hls_image, np.uint8([20, 200, 0]),
                              np.uint8([255, 255, 255]))  ## note that OpenCV uses BGR not RGB
    yellow_color = cv2.inRange(hls_image, np.uint8([10, 50, 100]), np.uint8([100, 255, 255]))

    combined_color_images = cv2.bitwise_or(white_color, yellow_color)
    return cv2.bitwise_and(image, image, mask=combined_color_images)


cap = cv2.VideoCapture('mp4/challenge.mp4')
prev_lane_lines = []

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    shape = frame.shape
    # cv2.imshow('video', frame)
    # gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Our operations on the frame come here
    image = color_selection(frame)

    sigma = 1
    gaussian_blur = cv2.GaussianBlur(image, (5, 5), sigma)

    aperature = 3
    canny = cv2.Canny(gaussian_blur, 50, 100, aperature)

    # apply region of interest
    vertices = np.array(
        [(shape[1] * 0.1, shape[0]), (shape[1] * 0.45, shape[0] * 0.6), (shape[1] * .55, shape[0] * 0.6),
         (shape[1], shape[0])], dtype=np.int32)
    roi_img = region_of_interest(canny, vertices=vertices)

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
    slopes_intercepts = avg_lines(alllines)
    # print slopes_intercepts
    endpoints = gen_endpoints(frame, slopes_intercepts)

    lane_lines = gen_lane_lines(frame, endpoints=find_mean_lines(endpoints, prev_lane_lines))

    final_img = weighted_img(lane_lines, frame)
    # print lane_lines
    # cv2.imshow('lane_lines', lane_lines)
    cv2.imshow('lanes', final_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
