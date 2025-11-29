import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# List for storing wrist positions
trail_points = []

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    h, w, c = frame.shape

    if results.pose_landmarks:
        wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        x = int(wrist.x * w)
        y = int(wrist.y * h)

        # Add point to trail
        trail_points.append((x, y))

        # Optional: Limit trail length
        if len(trail_points) > 100:
            trail_points.pop(0)

        # Draw main wrist point
        cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)

        # Draw motion trail (line between consecutive points)
        for i in range(1, len(trail_points)):
            cv2.line(frame, trail_points[i-1], trail_points[i], (0, 200, 255), 2)

    cv2.imshow("Wrist Tracking", frame)

    # ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
