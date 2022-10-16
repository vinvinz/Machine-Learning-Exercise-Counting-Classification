import cv2
import mediapipe as mp
import time
import pandas as pd
import numpy as np
import csv
import keyboard

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture('../videos/Sit-ups/situps5.mp4')

#array of variables containing file urls, labels etc.
vars = {
    "label": "Pushup_up",
    "recordID": 0,
    "csvFile": "test_dataset.csv",
    "mediaURL": ""
}

#Generate landmarks head row for CSV
def firstRow():
    landmarks = ['class']
    for val in range(1, 33+1):
        landmarks += ['x{}'.format(val), 'y{}'.format(val)]
    print(landmarks[1:])

#Function for appending data in csv
def writeCSV(csvFile, list):
    try:
        with open(csvFile, mode="a", newline='') as new_file:
                write_content = csv.writer(new_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                write_content.writerow(list)
    except Exception as e:
        print(e)
        pass

def export_landmark(result, label, recordID):
    try:
        keypoints = np.array([[res.x,res.y] for res in result.pose_landmarks.landmark]).flatten()
        keypoints = np.insert(keypoints, 0, label)
        print(keypoints)
        writeCSV(vars["csvFile"], keypoints)
    except Exception as e:
        print(e)
        pass
    
    

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    image = cv2.resize(image, (960, 540))
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        # landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        #Customized circle and connectors colors
        mp_drawing.DrawingSpec(color=(0,255,255), thickness=0, circle_radius=0),
        mp_drawing.DrawingSpec(color=(255,255,0), thickness=2, circle_radius=1))
    # Flip the image horizontally for a selfie-view display.
    if not results.pose_landmarks:
      continue    
    
    # If condition for triggering dataset capture
    if keyboard.is_pressed('r'):
        print("Landmarks Saved.")
        print(vars["label"])
        export_landmark(results, vars["label"], vars["recordID"])
    
    # print("X:",round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x, 4), "Y:", round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y, 4), "Z:", round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].z, 4), "V:", round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].visibility,4), "Ankle:", round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].visibility, 4))
    cv2.imshow('MediaPipe Pose', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
cv2.destroyAllWindows()