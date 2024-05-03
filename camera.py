import cv2

# Open camera with index 0
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Check if frame is valid
    if not ret or frame is None:
        print("Error: Failed to read frame from video stream.")
        break

    # Perform image processing tasks here...

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
