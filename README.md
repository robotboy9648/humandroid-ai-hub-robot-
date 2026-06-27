# ============================================================
# XYRON AI HUMANOID - RASPBERRY PI MASTER SYSTEM (V4)
# FEATURES: 24-Servo, LangChain Agent (Gemini), Full Body Track, 3D Sound
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
import numpy as np

# LangChain Essential Imports (Updated for Gemini)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_community.chat_message_histories import ConversationBufferWindowMemory

# ============================================================
# SETUPS (AI, SERVOS, SENSORS, SERIAL)
# ============================================================

GOOGLE_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# PCA9685 Setup
try:
    kit1 = ServoKit(channels=16, address=0x40) # Hands (10 Servos)
    kit2 = ServoKit(channels=16, address=0x41) # Face (4 Servos)
except Exception as e:
    print("Servo Kit Error:", e)

# Serial to ESP32
try: 
    esp32 = serial.Serial('/dev/ttyUSB0', 115200)
except: 
    print("ESP32 Not Connected")

# Sensors
try: 
    sensor = mpu6050(0x68) 
except: 
    print("MPU6050 Not Connected")

try:
    ultrasonic = DistanceSensor(echo=17, trigger=27)  
    front_ir_sensor = InputDevice(22)                 
    back_ir_sensor = InputDevice(23)                  
except Exception as e:
    print("Sensors GPIO Error:", e)

# AI Vision Setup
cap = cv2.VideoCapture(0)
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(min_detection_confidence=0.7)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Voice Setup
recognizer = sr.Recognizer()
engine = pyttsx3.init()
tts_lock = threading.Lock() # Thread-safety for pyttsx3

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
sound_angle = 0  
display_frame = None # For main thread OpenCV display

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def send_esp32(cmd):
    try: 
        esp32.write((cmd + "\n").encode())
    except: 
        pass

def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

# ============================================================
# LANGCHAIN TOOLS
# ============================================================
@tool
def move_robot_forward() -> str:
    """Robot ko aage chalane ya forward move karne ke liye is tool ka use karein."""
    global is_following, is_auto_walking, current_direction
    is_following, is_auto_walking, current_direction = False, True, "FORWARD"
    send_esp32("FORWARD")
    return "Main aage chal raha hoon."

@tool
def move_robot_backward() -> str:
    """Robot ko peeche le jaane ya backward move karne ke liye is tool ka use karein."""
    global is_following, is_auto_walking, current_direction
    is_following, is_auto_walking, current_direction = False, False, "BACKWARD"
    send_esp32("BACKWARD")
    return "Main peeche jaa raha hoon."

@tool
def stop_robot() -> str:
    """Robot ko rokne ya stop karne ke liye is tool ka use karein."""
    global is_following, is_auto_walking, current_direction
    is_following, is_auto_walking, current_direction = False, False, "STOP"
    send_esp32("STOP")
    return "Ruk gaya."

@tool
def start_robot_dance() -> str:
    """Robot ko dance karwane ke liye is tool ka upyog karein."""
    send_esp32("DANCE")
    return "Dancing mode on, chaliye dance karte hain!"

@tool
def follow_user() -> str:
    """User ko follow karne ya unke peeche chalne ke liye is tool ka use karein."""
    global is_following, is_auto_walking
    is_auto_walking = False; is_following = True
    return "Aapke peeche aa raha hoon."

robot_tools = [move_robot_forward, move_robot_backward, stop_robot, start_robot_dance, follow_user]

# ============================================================
# LANGCHAIN AGENT (Updated for Gemini)
# ============================================================
try:
    # Gemini Flash model fast processing ke liye best hai
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are Xyron, a highly powerful AI humanoid robot created by Mahendra Kushwaha (also known as Kushinagar Robot Boy). Speak short, natural Hindi/Hinglish. Use your tools whenever the user asks you to move, walk, dance, follow, or stop."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    robot_memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=5)
    
    # Modern tool calling agent for Gemini
    agent = create_tool_calling_agent(llm, robot_tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=robot_tools, verbose=False)
except Exception as e:
    print("LangChain Initialization Error:", e)

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
    print("XYRON:", text)
    with tts_lock:
        is_speaking = True
        threading.Thread(target=animate_speaking, daemon=True).start()
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print("Speech Engine Error:", e)
        is_speaking = False

def ask_langchain_agent(question):
    try:
        history = robot_memory.load_memory_variables({})["chat_history"]
        response = agent_executor.invoke({"input": question, "chat_history": history})
        output_text = response["output"]
        robot_memory.save_context({"input": question}, {"output": output_text})
        return output_text
    except Exception as e: 
        print("LangChain Error:", e)
        return "System logic me thoda issue hai."

# ============================================================
# VISION SYSTEM (Background Thread)
# ============================================================
def ai_vision_system():
    global head_x, head_y, is_following, is_speaking, current_direction, display_frame
    while True:
        ret, frame = cap.read()
        if not ret: 
            time.sleep(0.1)
            continue
            
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

        # Body Tracking
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

        # Update frame for Main Thread
        display_frame = frame
        time.sleep(0.01)

# ============================================================
# BACKGROUND SAFETY
# ============================================================
def obstacle_detection():
    global is_auto_walking, is_following, current_direction
    while True:
        try:
            front_distance = ultrasonic.distance * 100
            if front_distance < 25 or front_ir_sensor.value == 1:
                if current_direction == "FORWARD" or is_following or is_auto_walking:
                    send_esp32("STOP"); current_direction = "STOP"
                    is_auto_walking = False; is_following = False
                    speak("Aage kuch hai, main ruk gaya.")
            
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
        try: 
            kit2.servo[EYE_SERVO].angle = 40
            time.sleep(0.2)
            kit2.servo[EYE_SERVO].angle = 90
        except: pass
        time.sleep(5)

def track_sound_and_come():
    global sound_angle, is_auto_walking, current_direction
    speak("Awaaz ki disha mein aa raha hoon.")
    if 45 < sound_angle < 135: 
        send_esp32("RIGHT"); time.sleep(1)
    elif 225 < sound_angle < 315: 
        send_esp32("LEFT"); time.sleep(1)
    elif 135 <= sound_angle <= 225: 
        send_esp32("RIGHT"); time.sleep(2)
    is_auto_walking = True; current_direction = "FORWARD"; send_esp32("FORWARD")

# ============================================================
# VOICE COMMAND PROCESSOR
# ============================================================
def process_voice():
    while True:
        with sr.Microphone() as src:
            try:
                audio = recognizer.listen(src, timeout=5)
                cmd = recognizer.recognize_google(audio).lower()
                print("USER:", cmd)
                
                if "idhar aao" in cmd:
                    global is_following, is_auto_walking
                    is_auto_walking = False; is_following = False
                    track_sound_and_come()
                else: 
                    speak(ask_langchain_agent(cmd))
            except sr.WaitTimeoutError:
                pass
            except Exception as e:
                pass # Ignore random noise errors

# ============================================================
# START SYSTEM (Main Thread)
# ============================================================
if __name__ == "__main__":
    print("Starting Xyron AI with Gemini API Agent...")
    speak("Xyron System on. Ai hu , aadesh dijiye.")

    threading.Thread(target=ai_vision_system, daemon=True).start()
    threading.Thread(target=obstacle_detection, daemon=True).start()
    threading.Thread(target=balance_monitor, daemon=True).start()
    threading.Thread(target=eye_blink, daemon=True).start()
    threading.Thread(target=process_voice, daemon=True).start()

    # GUI must run on main thread to avoid crashes on Raspberry Pi
    while True:
        if display_frame is not None:
            cv2.imshow("XYRON V4", display_frame)
            if cv2.waitKey(1) == ord('q'): 
                break
        else:
            time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()
