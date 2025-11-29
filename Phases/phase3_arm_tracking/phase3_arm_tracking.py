import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, c = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Right arm joints
        shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
        wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

        # Convert to pixel coordinates
        s_x, s_y = int(shoulder.x * w), int(shoulder.y * h)
        e_x, e_y = int(elbow.x * w), int(elbow.y * h)
        w_x, w_y = int(wrist.x * w), int(wrist.y * h)

        # Draw joints
        cv2.circle(frame, (s_x, s_y), 8, (255, 0, 0), -1)   # Shoulder (blue)
        cv2.circle(frame, (e_x, e_y), 8, (0, 255, 0), -1)   # Elbow (green)
        cv2.circle(frame, (w_x, w_y), 8, (0, 0, 255), -1)   # Wrist (red)

        # Draw arm bones
        cv2.line(frame, (s_x, s_y), (e_x, e_y), (200, 200, 0), 4)
        cv2.line(frame, (e_x, e_y), (w_x, w_y), (200, 200, 0), 4)

        # Text display
        cv2.putText(frame, "Shoulder", (s_x + 10, s_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, "Elbow", (e_x + 10, e_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, "Wrist", (w_x + 10, w_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("Full Arm Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
