import cv2
import mediapipe as mp
import os
from collections import deque

# Create outputs folder
if not os.path.exists("../../outputs"):
    os.makedirs("../../outputs")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

video_path = "../../clips/clip1.mp4"
cap = cv2.VideoCapture(video_path)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
output_path = os.path.join("../../outputs", "right_arm_tracking_smooth.mp4")
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

# Buffers for smoothing (deque keeps last N frames)
N = 5
shoulder_buffer = deque(maxlen=N)
elbow_buffer = deque(maxlen=N)
wrist_buffer = deque(maxlen=N)

def smooth_point(buffer, new_point):
    buffer.append(new_point)
    x = int(sum(p[0] for p in buffer)/len(buffer))
    y = int(sum(p[1] for p in buffer)/len(buffer))
    return (x, y)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # Right arm points
        shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        elbow = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]
        wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST]

        # Convert to pixels
        s_pt = (int(shoulder.x * w), int(shoulder.y * h))
        e_pt = (int(elbow.x * w), int(elbow.y * h))
        w_pt = (int(wrist.x * w), int(wrist.y * h))

        # Smooth positions
        s_smooth = smooth_point(shoulder_buffer, s_pt)
        e_smooth = smooth_point(elbow_buffer, e_pt)
        w_smooth = smooth_point(wrist_buffer, w_pt)

        # Draw joints
        cv2.circle(frame, s_smooth, 8, (255,0,0), -1)
        cv2.circle(frame, e_smooth, 8, (0,255,0), -1)
        cv2.circle(frame, w_smooth, 8, (0,0,255), -1)

        # Draw arm
        cv2.line(frame, s_smooth, e_smooth, (0,255,255), 3)
        cv2.line(frame, e_smooth, w_smooth, (0,255,255), 3)

        # Labels
        cv2.putText(frame, "Shoulder", (s_smooth[0]+10, s_smooth[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
        cv2.putText(frame, "Elbow", (e_smooth[0]+10, e_smooth[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        cv2.putText(frame, "Wrist", (w_smooth[0]+10, w_smooth[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    # Write to output
    out.write(frame)

    cv2.imshow("Smoothed Right Arm Tracking", frame)
    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Smoothed video saved at {output_path}")
