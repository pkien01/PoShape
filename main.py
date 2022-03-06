import cv2
import os
import mediapipe as mp
import math
import random
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils 
mp_pose = mp.solutions.pose 


'''
def collinear(p1, p2, p3):
    print((p3[1] - p2[1])*(p2[0] - p1[0]))
    print((p2[1] - p1[1])*(p3[0] - p2[0]))
    return approx((p3[1] - p2[1])*(p2[0] - p1[0]),  (p2[1] - p1[1])*(p3[0] - p2[0]))
'''
def approx(a, b, pc=0.125): 
    return abs(a - b) < pc*abs(a + b)

def in_range(x, a, b, pc=0.05):
    return a*(1.-pc) <= x and x <= b*(1.+pc)

def angle(p1, p2, p3): # angle between p1-p2 and p1-p3
    res = abs(math.degrees(math.atan2(p3[1] - p1[1], p3[0] - p1[0]) - math.atan2(p2[1] - p1[1], p2[0] - p1[0])))
    if res > 180.: res = 360. - res
    return res

def collinear(p1, p2, p3):
    return approx(angle(p2, p1, p3), 180.)

def midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

def get_posture_shape(height, width, results):
    coord = results.pose_landmarks.landmark[0]
    nose = int(coord.x*width), int(coord.y*height)

    coord = results.pose_landmarks.landmark[11]
    left_shoulder = int(coord.x*width), int(coord.y*height)

    coord = results.pose_landmarks.landmark[12]
    right_shoulder = int(coord.x*width), int(coord.y*height)

    coord = results.pose_landmarks.landmark[13]
    left_elbow = int(coord.x*width), int(coord.y*height)

    coord = results.pose_landmarks.landmark[14]
    right_elbow = int(coord.x*width), int(coord.y*height)

    coord = results.pose_landmarks.landmark[15]
    left_wrist = int(coord.x*width), int(coord.y*height)

    coord = results.pose_landmarks.landmark[16]
    right_wrist = int(coord.x*width), int(coord.y*height)

    coord = results.pose_landmarks.landmark[23]
    left_hip = int(coord.x*width), int(coord.y*height)

    coord = results.pose_landmarks.landmark[24]
    right_hip = int(coord.x*width), int(coord.y*height)

    coord = results.pose_landmarks.landmark[25]
    left_knee = int(coord.x*width), int(coord.y*height)

    coord = results.pose_landmarks.landmark[26]
    right_knee = int(coord.x*width), int(coord.y*height)

    coord = results.pose_landmarks.landmark[27]
    left_ankle = int(coord.x*width), int(coord.y*height)

    coord = results.pose_landmarks.landmark[28]
    right_ankle = int(coord.x*width), int(coord.y*height)

    if (not collinear(left_shoulder, left_elbow, left_wrist)):
        print("Your left arm is not straight")
        return ""

    if (not collinear(right_shoulder, right_elbow, right_wrist)):
        print("Your right arm is not straight")
        return ""

    if (not collinear(left_hip, left_knee, left_ankle)):
        print("Your left leg is not straight")
        return ""
    
    if (not collinear(right_hip, right_knee, right_ankle)):
        print("Your right leg is not straight")
        return ""
    
    
    #arms_angle = angle(midpoint(left_shoulder, right_shoulder), left_wrist, right_wrist)
    left_arm_angle = angle(left_shoulder, midpoint(left_shoulder, right_shoulder), left_wrist)
    right_arm_angle = angle(right_shoulder, midpoint(left_shoulder, right_shoulder), right_wrist)
    legs_angle = angle(midpoint(left_hip, right_hip), left_ankle, right_ankle)
        
    print("angles:", left_arm_angle, right_arm_angle, legs_angle)
    if approx(left_arm_angle, 180.) and approx(right_arm_angle, 180.) and in_range(legs_angle, 40, 60):
        return "square"
    
    if in_range(left_arm_angle, 50., 80.) and in_range(right_arm_angle, 50., 80.) and in_range(legs_angle, 40, 60):
        return "triangle_up"
    
    if approx(left_arm_angle, 180.) and approx(right_arm_angle, 180.) and in_range(legs_angle, 0., 20.):
        return "triangle_down"
    
    if approx(left_arm_angle, 120.) and approx(right_arm_angle, 120.) and in_range(legs_angle, 40, 60):
        return "star"
    
    if in_range(left_arm_angle, 60., 100.) and in_range(right_arm_angle, 60., 100.) and in_range(legs_angle, 0., 20):
        return "stick"


def draw_pose(image):
    height, width, _ = image.shape
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks is None: return cv2.resize(image, (250, height*250//width)), ""
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    pos = get_posture_shape(height, width, results)
    #if pos != "": cv2.putText(image, pos, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)

    return cv2.resize(image, (250, height*250//width)), pos



cap = cv2.VideoCapture(0)
chosen_shape = random.randint(0, 4)
start_time = time.time()
wait_secs = 10.
score = 0
while(True):
    _, frame = cap.read()
    #cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('Camera', 2500, 1500)
    detected, pos_shape = draw_pose(frame) 

    #chosen_shape = random.randint(0, 5)
    
    shape_pic = 255 * np.ones((1500,1500,3), np.uint8)
    shape_pic[1000:(1000 + detected.shape[0]), 1000:(1000 + detected.shape[1]),:] = detected

    center = (shape_pic.shape[0]//2, shape_pic.shape[1] // 2)
    sz = 150
    match = False
    if chosen_shape == 0: #square
        cv2.rectangle(shape_pic, (center[0] - sz, center[0] - sz), (center[1] + sz, center[1] + sz), (0,0,255), -1)
        match = pos_shape == "square"

    elif chosen_shape == 1: #triangle up
        pts = np.array([(center[1], center[0] - sz), (center[1] - sz, center[0] + sz), (center[1] + sz, center[0] + sz)])
        cv2.drawContours(shape_pic, [pts], 0, (0,255,0), -1)
        match = pos_shape == "triangle_up"

    elif chosen_shape == 2: #triangle down
        pts = np.array([(center[1], center[0] + sz), (center[1] - sz, center[0] - sz), (center[1] + sz, center[0] - sz)])
        cv2.drawContours(shape_pic, [pts], 0, (255,0,0), -1)
        match = pos_shape == "triangle_down"

    elif chosen_shape == 3: #star
        star_img = cv2.imread("star.jpg")
        star_img = cv2.resize(star_img, (sz*2, sz*2))
        shape_pic[(center[0]-sz):(center[0]+sz), (center[1]-sz):(center[1]+sz)] = star_img 
        match = pos_shape == "star"

    elif chosen_shape == 4: #stick
        cv2.rectangle(shape_pic, (center[0] - sz//12, center[0] - sz), (center[1] + sz//12, center[1] + sz), (42, 42,165), -1)       
        match = pos_shape == "stick"

    cv2.putText(shape_pic, "Your score: " + str(score), (shape_pic.shape[0] - 300, 50), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 0), 3)
    if time.time() - start_time >= wait_secs:
        #wait_secs*math.exp(-score/15)
        if match:
            score += 1 
            cv2.putText(shape_pic, "CORRECT!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 100, 0), 3)
        else: cv2.putText(shape_pic, "WRONG!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 139), 3)

        time.sleep(1)
        start_time = time.time()

        chosen_shape = random.randint(0, 4)

    #cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('Camera', (2000, shape_pic.shape[1] * 2000 // shape_pic.shape[0]))
    cv2.imshow('Camera', cv2.resize(shape_pic, (800, (shape_pic.shape[1] * 800 // shape_pic.shape[0]))))
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()


'''
pic = cv2.imread("sample.jpg")
cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
cv2.resizeWindow('Camera', 800, 800)
pic = draw_pose(pic)
cv2.imshow('Camera', pic)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("sample_detected.jpg", pic)
'''