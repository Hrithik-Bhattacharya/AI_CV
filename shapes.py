import cv2
import numpy as np

def detect_shape(contour):
    # Approximate the contour
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    vertices = len(approx)

    if vertices == 3:
        return "Triangle"
    elif vertices == 4:
        # Check for square vs rectangle
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        return "Square" if 0.95 < aspect_ratio < 1.05 else "Rectangle"
    elif vertices > 4:
        return "Circle"
    else:
        return "Unknown"

# Load and preprocess image
img = cv2.imread("shapes.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop through contours and detect shapes
for contour in contours:
    shape = detect_shape(contour)
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
        cv2.putText(img, shape, (cx - 50, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# Show result
cv2.imshow("Detected Shapes", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
