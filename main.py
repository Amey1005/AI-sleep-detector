import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import os
import threading

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Eye landmark points
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Voice function
def speak_alert():

    os.system(
        'powershell -Command "& {Add-Type -AssemblyName System.Speech; '
        '(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'Wake up\')}"'
    )

# EAR function
def eye_aspect_ratio(eye_points, landmarks):

    p1 = np.array(landmarks[eye_points[0]])
    p2 = np.array(landmarks[eye_points[1]])
    p3 = np.array(landmarks[eye_points[2]])
    p4 = np.array(landmarks[eye_points[3]])
    p5 = np.array(landmarks[eye_points[4]])
    p6 = np.array(landmarks[eye_points[5]])

    vertical1 = distance.euclidean(p2, p6)
    vertical2 = distance.euclidean(p3, p5)

    horizontal = distance.euclidean(p1, p4)

    ear = (vertical1 + vertical2) / (2.0 * horizontal)

    return ear

# Webcam
cap = cv2.VideoCapture(0)

sleep_counter = 0
voice_alert = False

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)

    h, w, _ = frame.shape

    status = "AWAKE"
    color = (0, 255, 0)

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            landmarks = []

            for lm in face_landmarks.landmark:

                x = int(lm.x * w)
                y = int(lm.y * h)

                landmarks.append((x, y))

                cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)

            # Calculate EAR
            left_ear = eye_aspect_ratio(LEFT_EYE, landmarks)
            right_ear = eye_aspect_ratio(RIGHT_EYE, landmarks)

            ear = (left_ear + right_ear) / 2

            # Display EAR
            cv2.putText(frame,
                        f"EAR: {ear:.2f}",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2)

            # Sleep detection
            if ear < 0.25:

                sleep_counter += 1

                status = "DROWSY"
                color = (0, 165, 255)

            else:

                sleep_counter = 0
                voice_alert = False

            # Trigger sleep alert
            if sleep_counter > 5:

                status = "SLEEP DETECTED!"
                color = (0, 0, 255)

                # Speak only once
                if not voice_alert:

                    threading.Thread(
                        target=speak_alert,
                        daemon=True
                    ).start()

                    voice_alert = True

            # Status box
            cv2.rectangle(frame,
                          (20, 70),
                          (420, 140),
                          color,
                          -1)

            # Status text
            cv2.putText(frame,
                        status,
                        (35, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        3)

    # Title
    cv2.putText(frame,
                "AI Sleep Detector",
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2)

    # Show frame
    cv2.imshow("Sleep Detector", frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()