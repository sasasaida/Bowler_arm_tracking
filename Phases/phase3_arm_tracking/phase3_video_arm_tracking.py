import cv2
import mediapipe as mp

# Initialize pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Replace this with your video file path
video_path = "./clips/clip1.mp4"

cap = cv2.VideoCapture(video_path)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, c = frame.shape

    # Convert BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # Extract arm points
        shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        elbow = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]
        wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST]

        # Convert to pixel coordinates
        s_x, s_y = int(shoulder.x * w), int(shoulder.y * h)
        e_x, e_y = int(elbow.x * w), int(elbow.y * h)
        w_x, w_y = int(wrist.x * w), int(wrist.y * h)

        # Draw joints
        cv2.circle(frame, (s_x, s_y), 6, (255, 0, 0), -1)
        cv2.circle(frame, (e_x, e_y), 6, (0, 255, 0), -1)
        cv2.circle(frame, (w_x, w_y), 6, (0, 0, 255), -1)

        # Draw arm bones
        cv2.line(frame, (s_x, s_y), (e_x, e_y), (0, 255, 255), 3)
        cv2.line(frame, (e_x, e_y), (w_x, w_y), (0, 255, 255), 3)

    # Display video output
    cv2.imshow("Arm Tracking on Video", frame)

    if cv2.waitKey(20) & 0xFF == 27:  # slower playback
        break

cap.release()
cv2.destroyAllWindows()
