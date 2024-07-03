from controller import Robot, Keyboard
import cv2
import numpy as np

from colorize import laneDetector

# Create the Robot instance.
robot = Robot()

keyboard = robot.getKeyboard()
keyboard.enable(1000)

# Get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# Get left and right motors.
left_motor = robot.getDevice('left_motor')
right_motor = robot.getDevice('right_motor')

# Enable position control mode for motors.
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

# Set initial velocities.
left_velocity = 0.0
right_velocity = 0.0

detector = laneDetector()


# Get the camera device.
camera = robot.getDevice('camera')
camera.enable(timestep)

# Convert Webots image to OpenCV format
def convert_image(image, width, height):
    # Convert the Webots camera image to a NumPy array
    image_array = np.frombuffer(image, np.uint8).reshape((height, width, 4))
    # Convert BGRA to BGR
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_BGRA2BGR)
    return image_bgr


# Define LQR parameters
Kp = 0.03  # Proportional gain
Kd = 0.95  # Derivative gain

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    # Read keyboard input.
    key = keyboard.getKey()
    
    # Process keyboard input.
    if key == ord('W') or key == ord('w'):
        left_velocity = 5.0
        right_velocity = 5.0
    elif key == ord('S') or key == ord('s'):
        left_velocity = -5.0
        right_velocity = -5.0
    elif key == ord('A') or key == ord('a'):
        left_velocity = -2.5
        right_velocity = 2.5
    elif key == ord('D') or key == ord('d'):
        left_velocity = 2.5
        right_velocity = -2.5
    else:
        # Default behavior: Move forward
        left_velocity = 5.0
        right_velocity = 5.0
    
    # Read the camera image.
    image = camera.getImage()
    width = camera.getWidth()
    height = camera.getHeight()
    
    # Convert image to OpenCV format
    image_bgr = convert_image(image, width, height)
    
    # Crop the image to the bottom half
    cropped_image = image_bgr[height//2:, :]
    
    # Convert BGR image to HSV
    image_hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
    
    # Define HSV ranges for colors
    red_lower = np.array([0, 100, 100])   # Lower bound for red hue
    red_upper = np.array([10, 255, 255])  # Upper bound for red hue
    
    green_lower = np.array([0, 40, 40])
    green_upper = np.array([0, 0, 255])
    
    # Threshold the HSV image to get only the desired colors
    red_mask = cv2.inRange(image_hsv, red_lower, red_upper)
    green_mask = cv2.inRange(image_hsv, green_lower, green_upper)
    
    # Calculate moments of the red and green areas
    red_moments = cv2.moments(red_mask)
    green_moments = cv2.moments(green_mask)
    
    # laneDetector içindeki find_left_right_points fonksiyonunu kullanarak sol ve sağ noktaları bulun
    # left_point, right_point, _, _, _, _, _ = detector.find_left_right_points(image_bgr, draw=image_bgr.copy())
    
    new_image=detector.birdview_transform(image_bgr)
    red_detected = red_moments['m00'] > 0
    green_detected = green_moments['m00'] > 0
    
    # print(left_point,right_point)
    
    if red_detected:
        # Calculate the center of the red area
        red_cx = int(red_moments['m10'] / red_moments['m00'])
        # Calculate the error from the center of the image
        error = width / 2 - red_cx
    elif green_detected:
        # Calculate the center of the green area
        green_cx = int(green_moments['m10'] / green_moments['m00'])
        # Calculate the error from the center of the image
        error = green_cx - width / 2
    else:
        error = 0
    
    # Apply PID control
    left_velocity = 5.0 + Kp * error - Kd * left_velocity
    right_velocity = 5.0 - Kp * error - Kd * right_velocity
    
    # Set motor velocities.
    left_motor.setVelocity(left_velocity)
    right_motor.setVelocity(right_velocity)
    
    # Display the cropped image
    cv2.imshow("Cropped Camera View", cropped_image)
    cv2.imshow("Cropped Green View", green_mask)
    cv2.imshow("Cropped BirdView View", new_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
