import cv2

cap = cv2.VideoCapture('mp4/solidWhiteRight.mp4')

while (cap.isOpened()):

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    image = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)

    # Display the resulting frame
    cv2.imshow('frame', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
