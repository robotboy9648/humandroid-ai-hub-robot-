# ============================================================
# DHURANDHAR AI HUMANOID - RASPBERRY PI MASTER SYSTEM (V4)
# FEATURES: 24-Servo, ChatGPT, Full Body Track, 3D Sound, Front/Back Safety
# ============================================================

from adafruit_servokit import ServoKit
from mpu6050 import mpu6050
import cv2
import mediapipe as mp
import serial
import math
import time 
import threading
import speech_recognition as sr
import pyttsx3
import json
import os
from gpiozero import DistanceSensor, InputDevice
from openai import OpenAI
import numpy as np

# ============================================================
# SETUPS (AI, SERVOS, SENSORS, SERIAL)
# ============================================================
# OpenAI Setup
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY_HERY"
try: ai_client = OpenAI(api_key=OPENAI_API_KEY)
except: print("OpenAI Error!")

# PCA9685 Setup
kit1 = ServoKit(channels=16, address=0x40) # Kit 1: Hands (10 Servos)
kit2 = ServoKit(channels=16, address=0x41) # Kit 2: Face (4 Servos)

# Serial to ESP32
try: esp32 = serial.Serial('/dev/ttyUSB0', 115200)
except: print("ESP32 Not Connected")

# Sensors
try: sensor = mpu6050(0x68) # MPU6050 (Balance)
except: pass
ultrasonic = DistanceSensor(echo=17, trigger=27)  # Front Obstacle
front_ir_sensor = InputDevice(22)                 # Front Ground
back_ir_sensor = InputDevice(23)                  # Back Obstacle

# AI Vision Setup
cap = cv2.VideoCapture(0)
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(min_detection_confidence=0.7)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Voice Setup
recognizer = sr.Recognizer()
engine = pyttsx3.init()

# ============================================================
# PINS & GLOBAL VARIABLES
# ============================================================
# Face Servos
MOUTH_SERVO, EYE_SERVO, HEAD_UP_DOWN, HEAD_LEFT_RIGHT = 5, 6, 7, 8
# Right Hand Servos
R_SHOULDER_PITCH, R_SHOULDER_ROLL, R_ELBOW, R_WRIST, R_FINGERS = 0, 1, 2, 3, 4
# Left Hand Servos
L_SHOULDER_PITCH, L_SHOULDER_ROLL, L_ELBOW, L_WRIST, L_FINGERS = 5, 6, 7, 8, 9

head_x, head_y = 90, 90
is_following = False
is_auto_walking = False
is_speaking = False
current_direction = "STOP"
sound_angle = 0  # ReSpeaker se aane wala angle

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def send_esp32(cmd):
    try: esp32.write((cmd + "\n").encode())
    except: pass

def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

# ============================================================
# ANIMATION & SPEECH
# ============================================================
def animate_speaking():
    global is_speaking
    toggle = True
    while is_speaking:
        try:
            kit2.servo[MOUTH_SERVO].angle = 100 if toggle else 60
            kit1.servo[R_ELBOW].angle = 110 if toggle else 80
            kit1.servo[L_ELBOW].angle = 110 if toggle else 80
        except: pass
        toggle = not toggle
        time.sleep(0.15)
    try:
        kit2.servo[MOUTH_SERVO].angle = 90
        kit1.servo[R_ELBOW].angle = 90
        kit1.servo[L_ELBOW].angle = 90
    except: pass

def speak(text):
    global is_speaking
    print("DHURANDHAR:", text)
    is_speaking = True
    threading.Thread(target=animate_speaking, daemon=True).start()
    engine.say(text)
    engine.runAndWait()
    is_speaking = False

def ask_chatgpt(question):
    try:
        resp = ai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are Dhurandhar, an AI humanoid robot created by Mahendra Kushwaha (Kushinagar Robot Boy). Speak short Hindi/Hinglish."},
                {"role": "user", "content": question}
            ], max_tokens=100
        )
        return resp.choices[0].message.content.strip()
    except: return "Network connection mein problem hai."

# ============================================================
# VISION SYSTEM (Face + Body Mimic + Following)
# ============================================================
def ai_vision_system():
    global head_x, head_y, is_following, is_speaking, current_direction
    while True:
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        # Face Tracking
        face_res = face_detection.process(rgb)
        if face_res.detections:
            for det in face_res.detections:
                bbox = det.location_data.relative_bounding_box
                cx, cy = int((bbox.xmin + bbox.width/2)*w), int((bbox.ymin + bbox.height/2)*h)
                head_x = max(0, min(180, head_x - (cx - w//2)*0.03))
                head_y = max(0, min(180, head_y + (cy - h//2)*0.03))
                try:
                    kit2.servo[HEAD_LEFT_RIGHT].angle = head_x
                    kit2.servo[HEAD_UP_DOWN].angle = head_y
                except: pass

        # Body Tracking (Haath ki nakal)
        pose_res = pose.process(rgb)
        if pose_res.pose_landmarks and not is_speaking:
            lm = pose_res.pose_landmarks.landmark
            r_sh = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
            r_el = [lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x, lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
            r_wr = [lm[mp_pose.PoseLandmark.RIGHT_WRIST].x, lm[mp_pose.PoseLandmark.RIGHT_WRIST].y]
            r_hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP].x, lm[mp_pose.PoseLandmark.RIGHT_HIP].y]
            l_sh = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            l_el = [lm[mp_pose.PoseLandmark.LEFT_ELBOW].x, lm[mp_pose.PoseLandmark.LEFT_ELBOW].y]
            l_wr = [lm[mp_pose.PoseLandmark.LEFT_WRIST].x, lm[mp_pose.PoseLandmark.LEFT_WRIST].y]
            l_hip = [lm[mp_pose.PoseLandmark.LEFT_HIP].x, lm[mp_pose.PoseLandmark.LEFT_HIP].y]

            try:
                kit1.servo[R_SHOULDER_PITCH].angle = max(0, min(180, calculate_angle(r_hip, r_sh, r_el)))
                kit1.servo[R_ELBOW].angle = max(0, min(180, calculate_angle(r_sh, r_el, r_wr)))
                kit1.servo[L_SHOULDER_PITCH].angle = max(0, min(180, calculate_angle(l_hip, l_sh, l_el)))
                kit1.servo[L_ELBOW].angle = max(0, min(180, calculate_angle(l_sh, l_el, l_wr)))
            except: pass
            
            # Follow Me Logic
            if is_following:
                chest_x = (r_sh[0] + l_sh[0]) / 2 * w
                shoulder_width = abs(l_sh[0] - r_sh[0])
                if chest_x < w * 0.4: send_esp32("LEFT")
                elif chest_x > w * 0.6: send_esp32("RIGHT")
                else:
                    if shoulder_width < 0.2: 
                        send_esp32("FORWARD"); current_direction = "FORWARD"
                    elif shoulder_width > 0.4: 
                        send_esp32("STOP"); current_direction = "STOP"

        cv2.imshow("DHURANDHAR V4", frame)
        if cv2.waitKey(1) == ord('q'): break

# ============================================================
# BACKGROUND SAFETY & FEATURES
# ============================================================
def obstacle_detection():
    global is_auto_walking, is_following, current_direction
    while True:
        try:
            front_distance = ultrasonic.distance * 100
            
            # Aage ki Safety
            if front_distance < 25 or front_ir_sensor.value == 1:
                if current_direction == "FORWARD" or is_following or is_auto_walking:
                    send_esp32("STOP"); current_direction = "STOP"
                    is_auto_walking = False; is_following = False
                    speak("Aage kuch hai, main ruk gaya.")
            
            # Pichhe ki Safety
            if back_ir_sensor.value == 1: 
                if current_direction == "BACKWARD":
                    send_esp32("STOP"); current_direction = "STOP"
                    speak("Pichhe deewar hai, ruk gaya.")
        except: pass
        time.sleep(0.2)

def balance_monitor():
    while True:
        try:
            accel = sensor.get_accel_data()
            angle = math.atan2(accel['y'], accel['z']) * 180 / math.pi
            if abs(angle) > 25: send_esp32("BALANCE")
        except: pass
        time.sleep(0.1)

def eye_blink():
    while True:
        try: kit2.servo[EYE_SERVO].angle = 40; time.sleep(0.2); kit2.servo[EYE_SERVO].angle = 90
        except: pass
        time.sleep(5)

def track_sound_and_come():
    global sound_angle, is_auto_walking, current_direction
    speak("Awaaz ki disha mein aa raha hoon.")
    if 45 < sound_angle < 135: send_esp32("RIGHT"); time.sleep(1)
    elif 225 < sound_angle < 315: send_esp32("LEFT"); time.sleep(1)
    elif 135 <= sound_angle <= 225: send_esp32("RIGHT"); time.sleep(2)
    is_auto_walking = True; current_direction = "FORWARD"; send_esp32("FORWARD")

# ============================================================
# VOICE COMMAND PROCESSOR
# ============================================================
def process_voice():
    global is_following, is_auto_walking, current_direction
    while True:
        with sr.Microphone() as src:
            try:
                audio = recognizer.listen(src, timeout=5)
                cmd = recognizer.recognize_google(audio).lower()
                print("USER:", cmd)
                
                if "chalo" in cmd or "walk forward" in cmd:
                    is_following, is_auto_walking, current_direction = False, True, "FORWARD"
                    send_esp32("FORWARD"); speak("Main chal raha hoon.")
                
                elif "pichhe jao" in cmd or "go back" in cmd:
                    is_following, is_auto_walking, current_direction = False, False, "BACKWARD"
                    send_esp32("BACKWARD"); speak("Main pichhe jaa raha hoon.")
                
                elif "follow me" in cmd:
                    is_auto_walking = False; is_following = True; speak("Aapke pichhe aa raha hoon.")
                    
                elif "idhar aao" in cmd:
                    is_auto_walking = False; is_following = False; track_sound_and_come()
                
                elif "ruk jao" in cmd or "stop" in cmd:
                    is_following, is_auto_walking, current_direction = False, False, "STOP"
                    send_esp32("STOP"); speak("Ruk gaya.")
                    
                elif "dance" in cmd:
                    send_esp32("DANCE"); speak("Dancing mode on.")
                    
                else: 
                    speak(ask_chatgpt(cmd))
            except: pass

# ============================================================
# START SYSTEM
# ============================================================
print("Starting Dhurandhar AI...")
speak("Dhurandhar System on. Df ai hub, aadesh dijiye.")

threading.Thread(target=ai_vision_system, daemon=True).start()
threading.Thread(target=obstacle_detection, daemon=True).start()
threading.Thread(target=balance_monitor, daemon=True).start()
threading.Thread(target=eye_blink, daemon=True).start()
threading.Thread(target=process_voice, daemon=True).start()


while True: time.sleep(1)
