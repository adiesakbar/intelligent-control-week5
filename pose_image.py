from ultralytics import YOLO  # Import YOLO dari ultralytics

# Load model YOLOv8 Pose
model = YOLO("yolov8n-pose.pt")

# Deteksi pose pada gambar
results = model("https://ultralytics.com/images/bus.jpg")  # show=True tidak digunakan di sini

# Tampilkan hasil
for result in results:
    result.show()

# Simpan hasil
results[0].save(filename="pose_result.jpg")