import cv2
import mediapipe as mp
import os
import math

# ---------------- Paths ----------------
script_dir = os.path.dirname(os.path.abspath(__file__))
#video_path = os.path.join(script_dir, "..", "..", "clips", "clip1.mp4")
#output_dir = os.path.join(script_dir, "..", "..", "outputs")
video_path = r"C:\Users\l\OneDrive\Desktop\codes and projects\Projects\My Own\Cricket Bowler Arm Tracking System\clips\clip1.mp4"
output_dir = r"C:\Users\l\OneDrive\Desktop\codes and projects\Projects\My Own\Cricket Bowler Arm Tracking System\outputs"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "phase5_real_speed.mp4")

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
# Known real-world distance in meters (e.g., shoulder width ~0.45 m)
# Measure corresponding pixel distance in first frame (manual or automatic)
known_distance_m = 0.45  # meters
# Example: measure shoulder distance in pixels from first frame (we'll compute below)
meters_per_pixel = None

# ---------------- Main Loop ----------------
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

        # ---------------- Calibration: meters per pixel ----------------
        if meters_per_pixel is None:
            shoulder_pixel_dist = math.sqrt((s_x - lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w)**2 +
                                            (s_y - lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h)**2)
            meters_per_pixel = known_distance_m / shoulder_pixel_dist
            print(f"Calibrated meters per pixel: {meters_per_pixel:.5f} m/px")

        # ---------------- Pixel Velocity ----------------
        if prev_wrist is not None:
            dx = w_x - prev_wrist[0]
            dy = w_y - prev_wrist[1]
            pixel_distance = math.sqrt(dx*dx + dy*dy)
            pixel_speed = pixel_distance / time_per_frame  # px/s
            pixel_velocities.append(pixel_speed)

            # ---------------- Convert to real-world speed ----------------
            real_speed_m_s = pixel_speed * meters_per_pixel
            real_speed_kmh = real_speed_m_s * 3.6

            # Display on frame
            cv2.putText(frame, f"Speed: {real_speed_m_s:.2f} m/s | {real_speed_kmh:.2f} km/h",
                        (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        prev_wrist = (w_x, w_y)

    # Write and display
    out.write(frame)
    cv2.imshow("Phase 5 - Real Speed", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ Phase 5 video saved successfully at:", output_path)
