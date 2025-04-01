import cv2
import mediapipe as mp


# Initialize MediaPipe Solutions
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize Models
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5)
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Open the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for pose, hands, and face mesh detection
    pose_result = pose.process(rgb_frame)
    hands_result = hands.process(rgb_frame)

    # Draw pose landmarks if detected
    if pose_result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Draw hand landmarks if detected
    if hands_result.multi_hand_landmarks:
        for hand_landmarks in hands_result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the resulting frame
    cv2.imshow('Pose, Hands & Face Mesh Detector', frame)

    # Check for exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
