import cv2
import mediapipe as mp
import os

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
output_path = os.path.join("../../outputs", "right_arm_tracking_smooth_fast.mp4")
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

# Initialize smoothed points
s_smooth = None
e_smooth = None
w_smooth = None

alpha = 0.5  # smoothing factor (0.5 works well for fast motion)

def smooth(prev, curr, alpha=0.5):
    if prev is None:
        return curr
    x = int(alpha * curr[0] + (1-alpha) * prev[0])
    y = int(alpha * curr[1] + (1-alpha) * prev[1])
    return (x, y)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        elbow = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]
        wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST]

        # Convert to pixels
        s_pt = (int(shoulder.x * w), int(shoulder.y * h))
        e_pt = (int(elbow.x * w), int(elbow.y * h))
        w_pt = (int(wrist.x * w), int(wrist.y * h))

        # Exponential smoothing
        s_smooth = smooth(s_smooth, s_pt, alpha)
        e_smooth = smooth(e_smooth, e_pt, alpha)
        w_smooth = smooth(w_smooth, w_pt, alpha)

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

    # Write frame
    out.write(frame)
    cv2.imshow("Right Arm Tracking Smooth Fast", frame)
    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Smoothed fast video saved at {output_path}")
