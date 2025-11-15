import cv2
import sys
import os
from ultralytics import YOLO

class ModelObject:
    def __init__(self, model_path="farras_int8.onnx"):
        if not os.path.exists(model_path):
            # Fallback warning jika file onnx belum ada
            print(f"Warning: {model_path} tidak ditemukan. Pastikan file ada.")
            raise FileNotFoundError(f"Model file gak ketemu: {model_path}")
        
        print(f"Loading model: {model_path}...")
        self.model = YOLO(model_path, task='detect') 

        print("Class names:", self.model.names)

    def detect(self, frame):
        results = self.model.predict(
            source=frame,
            conf=0.78,
            iou=0.05,   
            verbose=False 
        )
        return results

if __name__ == "__main__":
    model_path = "farras_int8.onnx" 
    
    try:
        detector = ModelObject(model_path)
    except Exception as e:
        print(f"Error init model: {e}")
        sys.exit(1)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

    if not cap.isOpened():
        print("Webcam tidak terdeteksi.")
        sys.exit(1)

    print("Mulai deteksi... Tekan 'q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal mengambil frame.")
            break

        # --- PROSES DETEKSI ---
        # Langsung kirim frame OpenCV (BGR)
        results = detector.detect(frame)

        # --- GAMBAR HASIL ---
        # Ultralytics punya fungsi plot() bawaan 
        annotated_frame = results[0].plot()
        
        cv2.imshow("", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27: 
            break

    cap.release()
    cv2.destroyAllWindows()