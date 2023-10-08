import mediapipe as mp
import numpy as np
import cv2


def calculate_angle(a,b,c):
    a = np.array(a) #First
    b = np.array(b) #Mid
    c = np.array(c) #End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

fcc = cv2.VideoWriter_fourcc(*'XVID')
size = (1920, 1080)
video_number = 3
path = f"media/prolonged-standing/{video_number}"
video_output = cv2.VideoWriter(f"{path}.avi", fcc, 60, size)
cap = cv2.VideoCapture(f"{path}.mp4")
fps = 60
frame_count = 0
time = 0

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        result = pose.process(frame)
        landmarks = result.pose_landmarks.landmark

        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

        angle = calculate_angle(shoulder, hip, knee)

        if landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility > 0.3 or landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility > 0.3:
            if angle > 165:
                position = "Standing"
                frame_count += 1
            else:
                position = "Not Standing"
                frame_count = 0
        
        cv2.putText(frame, f"Position: {position}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)

        if frame_count > fps:
            time = int(frame_count/fps)
        else:
            time = 0
        
        cv2.putText(frame, f"Standing for: {time} seconds", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)

        if time > 30:
            cv2.putText(frame, "Please sit down on take a break", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)

        video_output.write(frame)
        cv2.imshow('Feed', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
    video_output.release()
    cap.release()
    cv2.destroyAllWindows()