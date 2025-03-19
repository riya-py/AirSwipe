import cv2 
import mediapipe as mp

# initialized mediapipe hand module
mp_hands=mp.solutions.hands
mp_draw=mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# start capturing video
cap=cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame=cap.read()
    if not ret:
        break

    # convert frame to rgb
    rgb_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results=hands.process(rgb_frame)

    # draw hand landmarks if detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # show the video
    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) & 0xff == 27:
        break

cap.release()
cv2.destroyAllWindows()
