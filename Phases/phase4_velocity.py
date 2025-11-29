import cv2
import mediapipe as mp
import os
import math

# Create outputs folder if it doesn't exist
if not os.path.exists("../../outputs"):
    os.makedirs("../../outputs")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

video_path = "./clips/clip1.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
# --- FIX for FPS=0 ---
if fps == 0 or fps is None:
    print("⚠ WARNING: FPS could not be read from video. Defaulting to 30 FPS.")
    fps = 30
time_per_frame = 1 / fps   # NEW ✔

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
output_path = os.path.join("./outputs", "phase4_velocity.mp4")
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

# --- NEW: Store previous wrist position ---
prev_wrist = None
pixel_velocities = []   # for analysis later

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # Right arm landmarks
        shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        elbow = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]
        wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST]

        # Convert normalized coords → pixels
        s_x, s_y = int(shoulder.x * w), int(shoulder.y * h)
        e_x, e_y = int(elbow.x * w), int(elbow.y * h)
        w_x, w_y = int(wrist.x * w), int(wrist.y * h)

        # Draw joints
        cv2.circle(frame, (s_x, s_y), 8, (255, 0, 0), -1)
        cv2.circle(frame, (e_x, e_y), 8, (0, 255, 0), -1)
        cv2.circle(frame, (w_x, w_y), 8, (0, 0, 255), -1)

        # Draw bones
        cv2.line(frame, (s_x, s_y), (e_x, e_y), (0, 255, 255), 3)
        cv2.line(frame, (e_x, e_y), (w_x, w_y), (0, 255, 255), 3)

        # Labels
        cv2.putText(frame, "Shoulder", (s_x+10, s_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
        cv2.putText(frame, "Elbow", (e_x+10, e_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        cv2.putText(frame, "Wrist", (w_x+10, w_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        # --- NEW: VELOCITY CALCULATION ---
        if prev_wrist is not None:
            dx = w_x - prev_wrist[0]
            dy = w_y - prev_wrist[1]
            pixel_distance = math.sqrt(dx*dx + dy*dy)

            pixel_speed = pixel_distance / time_per_frame   # pixels per second

            pixel_velocities.append(pixel_speed)

            # Display speed on frame
            cv2.putText(
                frame,
                f"Speed: {pixel_speed:.2f} px/s",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        prev_wrist = (w_x, w_y)  # update

    # Write frame to output
    out.write(frame)

    cv2.imshow("Phase 4 - Pixel Velocity", frame)
    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Phase 4 video with speed saved at {output_path}")
