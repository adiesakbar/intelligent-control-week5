from ultralytics import YOLO
import cv2

# Load model YOLOv8 Pose
model = YOLO("yolov8n-pose.pt")

# Inisialisasi kamera (0 untuk kamera default)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Deteksi pose menggunakan predict()
    results = model.predict(frame)

    # Tampilkan hasil
    for result in results:
        annotated_frame = result.plot()  # Tambahkan anotasi pada frame

        # Ambil keypoints (koordinat tubuh)
        keypoints = result.keypoints.xy if result.keypoints is not None else []

        # Deteksi tangan (biasanya index 9: pergelangan kanan, 10: pergelangan kiri)
        if len(keypoints) > 0:
            for kp in keypoints:
                left_wrist = kp[9] if len(kp) > 9 else None
                right_wrist = kp[10] if len(kp) > 10 else None

                # Tandai pergelangan tangan di frame
                if left_wrist is not None:
                    cv2.circle(annotated_frame, (int(left_wrist[0]), int(left_wrist[1])), 10, (0, 255, 0), -1)
                    cv2.putText(annotated_frame, "Left Hand", (int(left_wrist[0]), int(left_wrist[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                if right_wrist is not None:
                    cv2.circle(annotated_frame, (int(right_wrist[0]), int(right_wrist[1])), 10, (0, 0, 255), -1)
                    cv2.putText(annotated_frame, "Right Hand", (int(right_wrist[0]), int(right_wrist[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow("YOLOv8 Pose & Hand Detection", annotated_frame)

    # Keluar dari loop jika menekan tombol 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Membersihkan resource
cap.release()
cv2.destroyAllWindows()
