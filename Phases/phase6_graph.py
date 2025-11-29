import cv2
import mediapipe as mp
import os
import math
from collections import deque
import numpy as np


# ---------------- Paths ----------------
video_path = r"C:\Users\l\OneDrive\Desktop\codes and projects\Projects\My Own\Cricket Bowler Arm Tracking System\clips\clip1.mp4"
output_dir = r"C:\Users\l\OneDrive\Desktop\codes and projects\Projects\My Own\Cricket Bowler Arm Tracking System\outputs"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "phase6_trail_speed_graph.mp4")

print("Loading video from:", video_path)
print("Saving output to:", output_path)

# ---------------- Video Capture ----------------
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ ERROR: Video failed to open")
    exit()

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps is None:
    print("⚠ WARNING: FPS could not be read. Defaulting to 30")
    fps = 30
time_per_frame = 1 / fps

# ---------------- Video Writer ----------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
if not out.isOpened():
    print("❌ ERROR: VideoWriter failed")
    exit()

# ---------------- MediaPipe Setup ----------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, smooth_landmarks=True)

prev_wrist = None
pixel_velocities = []

# ---------------- Calibration ----------------
known_distance_m = 0.45  # meters (shoulder width)
meters_per_pixel = None

# ---------------- Trail Setup ----------------
trail_length = 15
wrist_trail = deque(maxlen=trail_length)

# ---------------- Speed Graph Setup ----------------
graph_width = 200
graph_height = 100
speed_history_length = 100
speed_history = deque(maxlen=speed_history_length)

# ---------------- Smoothing Setup ----------------
speed_smooth_window = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # Right arm landmarks
        shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        elbow = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]
        wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST]

        s_x, s_y = int(shoulder.x * w), int(shoulder.y * h)
        e_x, e_y = int(elbow.x * w), int(elbow.y * h)
        w_x, w_y = int(wrist.x * w), int(wrist.y * h)

        # Draw skeleton
        cv2.circle(frame, (s_x, s_y), 8, (255,0,0), -1)
        cv2.circle(frame, (e_x, e_y), 8, (0,255,0), -1)
        cv2.circle(frame, (w_x, w_y), 8, (0,0,255), -1)
        cv2.line(frame, (s_x, s_y), (e_x, e_y), (0,255,255), 3)
        cv2.line(frame, (e_x, e_y), (w_x, w_y), (0,255,255), 3)

        # ---------------- Calibration ----------------
        if meters_per_pixel is None:
            left_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
            shoulder_pixel_dist = math.sqrt((s_x - left_shoulder.x*w)**2 + (s_y - left_shoulder.y*h)**2)
            meters_per_pixel = known_distance_m / shoulder_pixel_dist
            print(f"Calibrated meters per pixel: {meters_per_pixel:.5f} m/px")

        # ---------------- Velocity ----------------
        speed_to_display = 0
        if prev_wrist is not None:
            dx = w_x - prev_wrist[0]
            dy = w_y - prev_wrist[1]
            pixel_distance = math.sqrt(dx*dx + dy*dy)
            pixel_speed = pixel_distance / time_per_frame
            pixel_velocities.append(pixel_speed)

            # Real-world speed
            real_speed_m_s = pixel_speed * meters_per_pixel
            real_speed_kmh = real_speed_m_s * 3.6

            # Smooth speed
            recent_speeds = pixel_velocities[-speed_smooth_window:]
            smooth_speed_m_s = sum([s*meters_per_pixel for s in recent_speeds]) / len(recent_speeds)
            smooth_speed_kmh = smooth_speed_m_s * 3.6
            speed_to_display = smooth_speed_kmh

            speed_history.append(smooth_speed_kmh)

            # Overlay speed text
            cv2.putText(frame, f"Speed: {smooth_speed_m_s:.2f} m/s | {smooth_speed_kmh:.2f} km/h",
                        (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        prev_wrist = (w_x, w_y)

        # ---------------- Wrist Trail ----------------
        wrist_trail.append((w_x, w_y))
        for i in range(1, len(wrist_trail)):
            cv2.line(frame, wrist_trail[i-1], wrist_trail[i], (0,0,255), 2)

    # ---------------- Speed Graph Overlay ----------------
    if speed_history:
        graph = 255 * np.ones((graph_height, graph_width, 3), dtype=np.uint8)
        max_speed = max(speed_history) * 1.2  # scale a bit higher
        min_speed = 0
        prev_point = None
        for i, speed in enumerate(speed_history):
            x = int(i * graph_width / speed_history_length)
            y = int(graph_height - (speed - min_speed) / (max_speed - min_speed) * graph_height)
            if prev_point is not None:
                cv2.line(graph, prev_point, (x, y), (0, 255, 0), 2)
            prev_point = (x, y)
        # Place graph on top-right corner
        frame[10:10+graph_height, w-graph_width-10:w-10] = graph

    # ---------------- Display & Write ----------------
    out.write(frame)
    cv2.imshow("Phase 6+ - Wrist Trail & Speed Graph", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ Phase 6+ video saved successfully at:", output_path)
