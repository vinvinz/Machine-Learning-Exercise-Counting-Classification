import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import pickle
import time
from landmarks import firstRow
from angle_calculator import calcAngle

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

reps_counter=0
reps_duration=0
current_pos=''
prev_pos=''
pTime = time.time()
cTime = 0
count_reset = True

cap = cv2.VideoCapture('../videos/Push-up/pushup8_flipped.mp4')


# '../videos/Sit-up/situps3.mp4'

with open('exercise.pkl', 'rb') as f:
    model = pickle.load(f)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    
    while cap.isOpened():
        success, image = cap.read()
        image = cv2.resize(image, (960, 540))
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = pose.process(image)

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
        
        landmarks = firstRow()
        
        try:
            row = np.array([[res.x,res.y] for res in results.pose_landmarks.landmark]).flatten()
            X=pd.DataFrame([row], columns=landmarks[1:])
            pose_classification = model.predict(X)[0]
            pose_prob = model.predict_proba(X)[0]
            
            if(pose_classification==1.0 and pose_prob[pose_prob.argmax()]>=.95):
                current_pos = "Situps Down"
                if(count_reset == True):
                    pTime = time.time()
                    count_reset = False
                if(prev_pos=="Situps UP" and count_reset == False):
                    reps_counter = reps_counter + 1
                    cTime = time.time()
                    reps_duration = cTime-pTime
                    count_reset = True
                    print("Prev time: ", pTime, "\nCurrent Time: ", cTime, "\nDifference: ", reps_duration)
            elif(pose_classification==2.0 and pose_prob[pose_prob.argmax()]>=.95):
                current_pos = "Situps UP"
            elif(pose_classification==3.0 and pose_prob[pose_prob.argmax()]>=.95):
                current_pos = "Pushups Down"
            elif(pose_classification==4.0 and pose_prob[pose_prob.argmax()]>=.95):
                current_pos = "Pushups UP"
                if(count_reset == True):
                    pTime = time.time()
                    count_reset = False
                if(prev_pos=="Pushups Down"):
                    reps_counter = reps_counter + 1  
                    cTime = time.time()
                    reps_duration = cTime-pTime  
                    count_reset = True
                    
            
                    
            #calculate angle for situps
            
            a = np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].z])
            b = np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].z])
            c = np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].z])
            
            angle = round(calcAngle(a, b, c),2)
            
            angle = "Angle: " + str(angle)
            
            reps_duration = round(reps_duration, 2)
            
            cv2.rectangle(image, (0,0), (250, 40), (245, 117, 16), -1)
            cv2.putText(image, current_pos
                        , (5,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(image, str(reps_counter)
                        , (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv2.LINE_AA)
            cv2.putText(image, str(reps_duration)
                        , (10,140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv2.LINE_AA)
            cv2.putText(image, str(angle)
                        , (10,210), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            prev_pos = current_pos
            
        except Exception as e:
            print(e)
            pass
        
        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()
    