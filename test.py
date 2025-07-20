import cv2
from ultralytics import YOLO

model = YOLO("c:\\Users\\GNC_LAB_5\\Documents\\Bodhini\\runs\\detect\\train33\\weights\\best.onnx")  
import cv2
import numpy as np

def add_sun_reflection(image, center_x, center_y, radius, intensity):
    """
    Adds a simulated sun reflection to an image.

    Args:
        image (numpy.ndarray): The input image.
        center_x (int): X-coordinate of the reflection center.
        center_y (int): Y-coordinate of the reflection center.
        radius (int): Radius of the main reflection circle.
        intensity (float): Intensity of the reflection (0.0 to 1.0).
    Returns:
        numpy.ndarray: The image with the added sun reflection.
    """
    reflection_mask = np.zeros(image.shape, dtype=np.uint8)

    # Draw the main reflection circle
    cv2.circle(reflection_mask, (center_x, center_y), radius, (255, 255, 255), -1)

    # Apply Gaussian blur to soften the reflection
    blurred_reflection = cv2.GaussianBlur(reflection_mask, (91, 91), 0) # Kernel size should be odd

    # Convert blurred_reflection to float for blending
    blurred_reflection_float = blurred_reflection.astype(np.float32) / 255.0

    # Blend with the original image
    output_image = cv2.addWeighted(image, 1.0, (blurred_reflection_float * 255).astype(np.uint8), intensity, 0)

    return output_image

# Load an image
img = cv2.imread('valid\\bw25.jpg') # Replace with your image path

if img is None:
    print("Error: Could not load image.")
else:
    # Add sun reflection at a specific location with desired intensity
    resized_img = cv2.resize(img, (640, 640))
    reflected_img = add_sun_reflection(resized_img, 500, 300, 100, 0.7)


results = model(reflected_img,iou=0.6,conf=0.1)     
annotated_img = results[0].plot()  
resized_output = cv2.resize(annotated_img, (640, 640))


cv2.imshow('YOLO Detections', resized_output)
cv2.waitKey(0)
cv2.destroyAllWindows()
