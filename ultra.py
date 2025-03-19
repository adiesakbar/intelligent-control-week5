from ultralytics import YOLO
import cv2
import numpy as np

# Load model YOLOv8 Pose
model = YOLO("yolov8n-pose.pt")

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

# Indeks titik kunci berdasarkan model YOLOv8 Pose
KEYPOINTS_TO_KEEP = {
    0: "Hidung",
    1: "Mata Kiri",
    2: "Mata Kanan",
    3: "Telinga Kiri",
    4: "Telinga Kanan",
    5: "Bahu Kiri",
    6: "Bahu Kanan",
    7: "Siku Kiri",
    8: "Siku Kanan",
    9: "Pergelangan Tangan Kiri",
    10: "Pergelangan Tangan Kanan",
    11: "Pinggul Kiri",
    12: "Pinggul Kanan",
    13: "Lutut Kiri",
    14: "Lutut Kanan",
    15: "Pergelangan Kaki Kiri",
    16: "Pergelangan Kaki Kanan"
}

# Definisi koneksi antara titik kunci untuk membentuk pose tubuh
POSE_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Kepala
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Lengan kiri dan kanan
    (5, 11), (6, 12), (11, 12),  # Badan atas
    (11, 13), (13, 15), (12, 14), (14, 16),  # Kaki kiri dan kanan
    (5, 6)  # Bahu kiri ke bahu kanan
]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Konversi frame ke RGB untuk YOLOv8
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Deteksi pose
    results = model(frame_rgb)
    
    for result in results:
        annotated_frame = frame.copy()
        
        if result.keypoints is not None:  # Pastikan ada keypoints yang terdeteksi
            keypoints = result.keypoints.xy  # Ambil koordinat titik kunci
            
            # Gambar titik kunci
            for i, name in KEYPOINTS_TO_KEEP.items():
                if i < len(keypoints[0]):  # Pastikan indeks valid
                    x, y = int(keypoints[0][i][0]), int(keypoints[0][i][1])
                    cv2.circle(annotated_frame, (x, y), 5, (0, 255, 0), -1)  # Titik hijau
                    cv2.putText(annotated_frame, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Gambar koneksi antar titik untuk membentuk pose
            for (p1, p2) in POSE_CONNECTIONS:
                if p1 < len(keypoints[0]) and p2 < len(keypoints[0]):
                    x1, y1 = int(keypoints[0][p1][0]), int(keypoints[0][p1][1])
                    x2, y2 = int(keypoints[0][p2][0]), int(keypoints[0][p2][1])
                    cv2.line(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Garis biru
        
        cv2.imshow("YOLOv8 Pose Estimation", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
