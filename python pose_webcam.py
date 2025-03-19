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

    cv2.imshow("YOLOv8 Pose Estimation", annotated_frame)

    # Keluar dari loop jika menekan tombol 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Membersihkan resource
cap.release()
cv2.destroyAllWindows()