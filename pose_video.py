from ultralytics import YOLO
import cv2
import os

# Debugging: Tampilkan direktori saat ini
print("Current Directory:", os.getcwd())

# Path video (gunakan path absolut jika perlu)
video_path = "IMG_7351.MOV"

# Periksa apakah file video ada sebelum diproses
if not os.path.exists(video_path):
    print(f"Error: File '{video_path}' tidak ditemukan. Pastikan nama dan lokasi file benar.")
    exit()

# Load model YOLOv8 Pose
model = YOLO("yolov8n-pose.pt")

# Buka video dengan OpenCV
cap = cv2.VideoCapture(video_path)

# Periksa apakah video berhasil dibuka
if not cap.isOpened():
    print("Error: Gagal membuka video.")
    exit()

# Ambil informasi video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

# Buat objek writer untuk menyimpan video hasil deteksi
out = cv2.VideoWriter("output_pose.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Keluar jika sudah selesai

    # *Membalik frame jika video terbalik*
    frame = cv2.rotate(frame, cv2.ROTATE_180)  # Balik 180 derajat jika video terbalik
    # Jika hanya terbalik secara horizontal/vertikal, gunakan:
    # frame = cv2.flip(frame, 0)  # Flip secara vertikal
    # frame = cv2.flip(frame, 1)  # Flip secara horizontal

    # Deteksi pose
    results = model(frame)

    # Tambahkan anotasi pada frame
    for result in results:
        annotated_frame = result.plot()

    # Simpan frame ke video output
    out.write(annotated_frame)

    # Tampilkan hasil
    cv2.imshow("YOLOv8 Pose Estimation", annotated_frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bebaskan sumber daya
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video hasil deteksi telah disimpan sebagai 'output_pose.mp4'.")