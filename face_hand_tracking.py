import cv2
import mediapipe as mp

# Inisialisasi MediaPipe Face Mesh dan Hands
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Konfigurasi Face Mesh dan Hand Tracking
face_mesh = mp_face_mesh.FaceMesh()
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Buka kamera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    # Konversi frame ke RGB untuk MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Proses Face Mesh
    face_results = face_mesh.process(frame_rgb)
    
    # Proses Hand Tracking
    hand_results = hands.process(frame_rgb)
    
    # Gambar Face Mesh
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
    
    # Gambar Hand Tracking
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2)
            )
    
    # Tampilkan hasil
    cv2.imshow('Face Mesh & Hand Tracking', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
