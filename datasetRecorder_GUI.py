import tkinter as tk 
from tkinter import *
import customtkinter as ck 
from tkinter import ttk, filedialog
from tkinter.filedialog import askopenfile
import os

import cv2
import mediapipe as mp
import time
import pandas as pd
import numpy as np

from record_dataset import writeCSV, export_landmark

from PIL import Image, ImageTk 

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_tracking_confidence=0.8, min_detection_confidence=0.8)

      
class datasetGUI:
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry("720x540")
        
        self.csv_filepath = ""
        self.dataset_label = ""
        
        self.frame = tk.Frame(height=380, width=720)
        self.frame.place(x=0, y=0)
        self.lmain = tk.Label(self.frame) 
        self.lmain.place(x=0, y=0)
        
        self.utilityComponents()
        
        self.root.mainloop() 
        
    def open_file(self):
        file = filedialog.askopenfile(mode='r', filetypes=[('All files', '*.*')])
        if file:
            filepath = os.path.relpath(file.name)
            Label(self.root, text="The File is located at : " + str(filepath), font=('Aerial 11')).place(x=0, y=0)
            self.openVideo(filepath)
            
    def open_csv(self):
        file = filedialog.askopenfile(mode='r', filetypes=[('All files', '*.*')])
        if file:
            filepath = os.path.relpath(file.name)
            self.csv_filepath = filepath
            Label(self.root, text="File selected : " + str(filepath), font=('Aerial 11')).place(x=220, y=430)
            
    def record_landmarks(self, result, label, csvFilePath):
        print("Recording Landmarks...")
        try:
            label = float(label)
            export_landmark(result, label, csvFilePath)
        except Exception as e:
            print(e)
            pass
        
            
    def utilityComponents(self):
        ttk.Button(self.root, text="Browse", command=self.open_file).place(x=120, y=390)
        label = Label(self.root, text="Open Video File:", font=('Arial 11'))
        label.place(x=0, y=390)
        
        
    def openVideo(self, vidpath):
        cap = cv2.VideoCapture(vidpath)
        
        label = Label(self.root, text="Choose CSV File:", font=('Arial 11'))
        label.place(x=0, y=430)
        ttk.Button(self.root, text="CSV Location", command=self.open_csv).place(x=130, y=430)
        
        Label(self.root, text="Row Label: ", font=('Arial 11')).place(x=0, y=470)
        self.dataset_label = tk.Entry(self.root, width=10, textvariable = 0)
        self.dataset_label.place(x=100, y=470)
        
        
        def detect():
            ret, frame = cap.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
            image = cv2.resize(image, (720, 380))
            self.results = pose.process(image)
            mp_drawing.draw_landmarks(image, self.results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                mp_drawing.DrawingSpec(color=(0,255,255), thickness=0, circle_radius=0), 
                mp_drawing.DrawingSpec(color=(255,255,0), thickness=2, circle_radius=1)) 
            img = image[:, :960, :] 
            imgarr = Image.fromarray(img) 
            imgtk = ImageTk.PhotoImage(imgarr) 
            self.lmain.imgtk = imgtk 
            self.lmain.configure(image=imgtk)
            self.lmain.after(10, detect) 
        
        ttk.Button(self.root, text="Record Data", command=lambda: self.record_landmarks(result=self.results, label=self.dataset_label.get(), csvFilePath=self.csv_filepath)).place(x=0, y=490)    
        
        detect()        
        # with mp_pose.Pose(
        #         min_detection_confidence=0.8,
        #         min_tracking_confidence=0.8) as pose:
        #     while cap.isOpened():
        #         success, image = cap.read()
        #         if not success:
        #             print("Ignoring empty camera frame.")
        #             # If loading a video, use 'break' instead of 'continue'.
        #             continue
        #         ret, frame = cap.read()
        #         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #         image = cv2.resize(image, (720, 380))
        #         results = pose.process(image)
        #         mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
        #             mp_drawing.DrawingSpec(color=(0,255,255), thickness=0, circle_radius=0), 
        #             mp_drawing.DrawingSpec(color=(255,255,0), thickness=2, circle_radius=1)) 

        #         img = image[:, :720, :] 
        #         imgarr = Image.fromarray(img) 
        #         imgtk = ImageTk.PhotoImage(imgarr) 
        #         self.lmain.imgtk = imgtk 
        #         self.lmain.configure(image=imgtk)
        #         self.lmain.after(10, self.openVideo(vidpath=filepath))
                
                
datasetGUI()