import cv2
import numpy as np

import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('./perpective.png')

# Define the points for the perspective transform
# These should be the corners of the quadrilateral you want to transform
# Points should be in the following order: top-left, top-right, bottom-right, bottom-left
pts1 = np.float32([[160, 300], [480, 300], [0, 400], [640, 400]])

# Define the points for the destination perspective
# Typically, this will be a rectangle
pts2 = np.float32([[0, 0], [640, 0],  [0, 480],[640, 480]])

# Compute the perspective transformation matrix
matrix = cv2.getPerspectiveTransform(pts1, pts2)

# Apply the perspective transformation
result = cv2.warpPerspective(image, matrix, (640, 480))

# Display the original and transformed images


# Plot the original image with the selected points
plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.imshow(image)
plt.scatter(pts1[:, 0], pts1[:, 1], c='red', marker='o')
plt.title('Original Image with Selected Points')
for i, txt in enumerate(['Point 1', 'Point 2', 'Point 3', 'Point 4']):
    plt.annotate(txt, (pts1[i, 0], pts1[i, 1]), color='white', fontsize=12, ha='center')

# Plot the transformed image
plt.subplot(122)
plt.imshow(result)
plt.title('Perspective Transform')

plt.show()
