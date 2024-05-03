import cv2
import numpy as np
import pandas as pd

# Load color ranges from CSV file
color_data = pd.read_csv('color_name.csv')

# Initialize video capture object (assuming camera index 0)
cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("Error: Failed to read frame from video stream.")
            break

        # Convert frame to HSV color space
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Iterate over color ranges
        for _, (color_name, h_min, s_min, v_min, h_max, s_max, v_max) in color_data.iterrows():
            # Create lower and upper bounds for the color range
            lower_bound = np.array([h_min, s_min, v_min])
            upper_bound = np.array([h_max, s_max, v_max])

            # Create mask for current color range
            mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Sort contours by area in descending order
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            # Iterate over contours to check significance
            for contour in contours:
                # Calculate area of contour
                area = cv2.contourArea(contour)

                # Check if contour area is above a certain threshold (e.g., 100 pixels)
                if area > 100:
                    # Draw rectangle and display color label for significant contour
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, color_name.capitalize(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Break out of the loop to process the next color
                    break

        # Display processed frame
        cv2.imshow('Frame', frame)

        # Break loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release video capture object and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
