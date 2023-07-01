# Exercise Counting Classification ML Model
A machine learning model capable of classifying 5 calisthenics exercises namely; Push-ups, Sit-ups, Squats, Jumping-jacks, and Planking using Mediapipe Pose.
### This repository contains Scripts and GUI Applications that can be used to:
>- Record a dataset using GUI App
>- Script for Training Machine Learning Model
>- Test the performance of the Model using Mediapipe Pose and opening a sample video
>- Measure the angle of 3 points in 3 Dimensions using the X, Y, and Z values from Mediapipe Pose Landmarks

  
![Exercise-Classification-Preview-Recovered-2](https://github.com/vinvinz/Machine-Learning-Exercise-Counting-Classification/assets/80497740/c942497d-2e01-4ef0-ab5e-9cbbcb60ac82)

### Dataset
The Dataset used in training the Model contains 9 labels:
| Labels | Exercises |
|-----:|-----------|
|     1| Situps Down |
|     2| Situps Up |
|     3| Pusups Down |
|     4| Pushups Up |
|     5| Plank |
|     6| Squat Up |
|     7| Squat Down |
|     8| Jumping Jack Up |
|     9| Jumping Jack Down |

### Model Performance
The score of the latest ML Model (exercisev3.pkl):
![Model Evaluation](https://github.com/vinvinz/Machine-Learning-Exercise-Counting-Classification/assets/80497740/f78b2d90-64da-4412-b8ad-ed6c8b227889)

The Confusion Matrix produced through testing the prediction of the ML Model:
![ML Model Confusion Matrix](https://github.com/vinvinz/Machine-Learning-Exercise-Counting-Classification/assets/80497740/2951eb7a-fc98-4939-bc48-94664b1b77ae)

### How to use the repository
1. Create a Python virtual environment
2. Activate the Python environment
3. Install the dependencies in requirements.txt
   >pip install requirements.txt
4. Create your own Dataset using "datasetRecorder_GUI.py" GUI App
5. Train your model using the dataset or CSV file using "train_model.py"
6. Test your model using "exercise_test_counter.py"
