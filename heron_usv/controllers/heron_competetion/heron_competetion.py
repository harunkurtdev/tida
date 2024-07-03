from controller import Robot, Keyboard, Compass
import cv2
import numpy as np
import heapq
import threading
import math
from multiprocessing import Queue, Process


from pid import PID

# PID denetleyici parametreleri
kp = 0.1
ki = 0
kd = 0.05
target_angle = 278.0

pid_controller = PID(kp, ki, kd, target_angle)

pid_path=PID(kp, ki, kd, 0)
robot = Robot()

keyboard = robot.getKeyboard()
keyboard.enable(1000)

timestep = int(robot.getBasicTimeStep())

compass = robot.getDevice('compass')
compass.enable(timestep)

left_motor = robot.getDevice('left_motor')
right_motor = robot.getDevice('right_motor')

left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

# Set initial velocities.
left_velocity = 0.0
right_velocity = 0.0

Kp = 0.03  
Ky = 0.0000  
Kd = 0.95  

Kp_forward=0.5
Kd_forward=0.5
Kp_turn=0.5
forward_velocity=0


start = (320, 480)
goal = (320, 0)

lock = threading.Lock()
path = []


camera = robot.getDevice('camera')
camera.enable(timestep)

def convert_image(image, width, height):
    image_array = np.frombuffer(image, np.uint8).reshape((height, width, 4))
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_BGRA2BGR)
    return image_bgr

src_points = np.float32([[160, 250], [480, 250], [0, 480], [640, 480]])
dst_points = np.float32([[0, 0], [640, 0],  [0, 480],[640, 480]])
M = cv2.getPerspectiveTransform(src_points, dst_points)

def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def astar(start, goal, potential_field):
    start = tuple(start)
    goal = tuple(goal)
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []
    
    heapq.heappush(oheap, (fscore[start], start))
    
    while oheap:
        current = heapq.heappop(oheap)[1]
        
        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data[::-1]
        
        close_set.add(current)
        neighbors = [(current[0] + i, current[1] + j) for i, j in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
        
        for neighbor in neighbors:
            if 0 <= neighbor[0] < potential_field.shape[1] and 0 <= neighbor[1] < potential_field.shape[0]:
                if potential_field[neighbor[1], neighbor[0]] == 255:
                    continue
            else:
                continue
            
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue
            
            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    
    return []

def draw_path_on_image(image, path):
    result_image = image.copy()
    
    for point in path:
        cv2.circle(result_image, point, radius=2, color=(0, 0, 255), thickness=5) 
        
    return result_image

def draw_trajectory_on_image(image, path):
    result_image = image.copy()
    
    if len(path) > 0:
        prev_point = path[0]
        for point in path[1:]:
            cv2.line(result_image, prev_point, point, color=(0, 255, 0), thickness=5)  
            prev_point = point
    
    return result_image
    

image=np.zeros((640,480,4),np.uint8)

height=480
width=640
collision_potential_field=np.zeros((640,480,4),np.uint8)
image_bgr=np.zeros((640,480,4),np.uint8)

def process_image_and_plan_path():
    global path,M,start,goal
    
    global image,width,height,collision_potential_field,image_bgr
    while robot.step(timestep) != -1:
    
        image_bgr = convert_image(image, width, height)

        birdview_image = cv2.warpPerspective(image_bgr, M, (640, 480))
        hsv = cv2.cvtColor(birdview_image, cv2.COLOR_BGR2HSV)
        
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask2
        
        lower_green = np.array([35, 100, 100])  
        upper_green = np.array([85, 255, 255]) 

        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        combined_mask = red_mask + yellow_mask + green_mask
        
        # cv2.imshow("combined_mask",combined_mask)
        obstacle_kernel = np.ones((10, 10), np.uint8)
        dilated_combined_mask = cv2.dilate(combined_mask, obstacle_kernel, iterations=2)
        
        potential_field = np.zeros((height, width), dtype=np.float32)
        potential_field[dilated_combined_mask > 0] = 255
        # cv2.imshow("potential_field",potential_field)
        vehicle_radius = 64
        vehicle_kernel = np.ones((vehicle_radius * 2, vehicle_radius * 2), np.uint8)
        collision_potential_field = cv2.dilate(potential_field, vehicle_kernel, iterations=1)
                
      
        cv2.imshow("image collision potential",collision_potential_field)
        path_trajectory = np.array_split(path, 8)
        try:
            for point in np.array_split(path, 40):
                cv2.circle(birdview_image, point[-1], radius=2, color=(0, 0, 255), thickness=5)  # Kırmızı renk, 2 piksel çapında
            
                # next_point = path[0]
                
            for point_track in path_trajectory:
                cv2.circle(birdview_image, point_track[-1], radius=2, color=(0, 255, 255), thickness=5)  # Kırmızı renk, 2 piksel çapında
            
            
        except Exception as e:
            print(e)
        
        cv2.imshow("image_bgr",birdview_image)
        
        # cv2.imwrite("perpective.png",image_bgr)       
        # cv2.imshow("image collision potential",collision_potential_field)            
        # cv2.imshow("Image with Path and Trajectory", image_with_trajectory)
        
    

def robot_control():
    global path,forward_velocity,image_bgr,collision_potential_field
    
    global image,width,height
    
    while robot.step(timestep) != -1:
        values = compass.getValues()
        north = values[0]
        east = values[1]
        up = values[2]
        
        direction = math.atan2(east, north)
        if direction < 0.0:
            direction += 2 * math.pi
        
        degrees = math.degrees(direction)
        key = keyboard.getKey()
        
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
            left_velocity = 1.0
            right_velocity = 1.0

        
        center_coordinates = (320, 480) 

        if path:
            path_trajectory = np.array_split(path, 8)
            
            # print(path_trajectory[-1][1])
            # print(path_trajectory[1])            
            error_x = camera.getWidth() / 2 - path_trajectory[2][-1][0]
            error_y = camera.getHeight() / 2 - path_trajectory[2][-1][1]
            
            direction_path = math.atan2(path_trajectory[2][-1][0],path_trajectory[2][-1][1])
            if direction_path < 0.0:
                direction_path += 2 * math.pi
            degrees_path = math.degrees(direction_path)
            correction_path = pid_path.compute(degrees_path)
            
            correction = pid_controller.compute(degrees)
            print(direction_path,degrees_path,correction_path)
            error_x=correction_path
            error_y=-5

            
            forward_velocity = 1.0 - Kp_forward * error_y - Kd_forward * forward_velocity
            turn_velocity = -Kp_turn * error_x

            left_velocity =  turn_velocity + correction
            right_velocity = turn_velocity - correction

            # if abs(error_x) < 5 and abs(error_y) < 5:
                # path.pop(0)
        else:
            left_velocity = 0
            right_velocity = 0
            
        left_motor.setVelocity(left_velocity)
        right_motor.setVelocity(right_velocity)
        
def process_plan_path():
    global path,collision_potential_field,start, goal
    
    while robot.step(timestep) != -1:
        try:
           path = astar(start, goal, collision_potential_field)
           print(path)
           if len(path)==0:
                col_index=0
                for row_index in range(640):
                    if collision_potential_field[row_index, col_index] == 0:
                        goal=(row_index,0)
                        print(goal)
                        break
               
        except Exception as e:
            print(e)

def main():
    global image,width,height
    thread = threading.Thread(target=process_image_and_plan_path)
    thread.start()
    
    thread1 = threading.Thread(target=process_plan_path)
    thread1.start()
    
    robot_con = threading.Thread(target=robot_control)
    robot_con.start()
    
    while robot.step(timestep) != -1:
    
        image = camera.getImage()
        width = camera.getWidth()
        height = camera.getHeight()
        
        cv2.waitKey(25)
    
        pass

    thread.join()
    thread1.join()
    robot_con.join()
if __name__ == "__main__":
    main()


