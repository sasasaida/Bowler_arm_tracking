import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize drawing utils (to draw keypoints)
mp_drawing = mp.solutions.drawing_utils

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run pose detection
    results = pose.process(rgb)

    # If pose landmarks found
    if results.pose_landmarks:
        # Get RIGHT WRIST landmark (ID = 16)
        wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

        h, w, c = frame.shape

        # Convert normalized coords → pixel coords
        wrist_x = int(wrist.x * w)
        wrist_y = int(wrist.y * h)

        # Draw a circle on wrist
        cv2.circle(frame, (wrist_x, wrist_y), 8, (0, 255, 0), -1)

        # Print coordinates
        cv2.putText(frame, f"Wrist: {wrist_x}, {wrist_y}",
                    (wrist_x + 10, wrist_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show output
    cv2.imshow("Wrist Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
