import os
import sys
import cv2
import numpy as np
import torch

# 1. Định nghĩa đường dẫn tới thư mục SilentFaceAntiSpoofing
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SILENT_FACE_DIR = os.path.join(CURRENT_DIR, "SilentFaceAntiSpoofing")

# Thêm SilentFaceAntiSpoofing vào sys.path để các file bên trong import được 'src'
if SILENT_FACE_DIR not in sys.path:
    sys.path.insert(0, SILENT_FACE_DIR)

# 2. Import trực tiếp từ thư viện SilentFace
from SilentFaceAntiSpoofing.src.anti_spoof_predict import AntiSpoofPredict
from SilentFaceAntiSpoofing.src.generate_patches import CropImage


class LivenessDetector:
    def __init__(self, device_id=0):
        self.device_id = device_id
        # Khởi tạo predictor và cropper từ SilentFace
        self.model_test = AntiSpoofPredict(device_id)
        self.cropper = CropImage()

    def check_liveness(self, img, bbox):
        """
        img: Ảnh BGR đọc bằng OpenCV
        bbox: Vùng khuôn mặt [x, y, w, h] từ InsightFace
        Trả về: (is_real: bool, confidence: float)
        """
        try:
            # Đường dẫn file mô hình SilentFace
            model_dir = os.path.join(SILENT_FACE_DIR, "resources", "anti_spoof_models")
            model_name = "2.7_80x80_MiniFASNetV2.pth"
            model_path = os.path.join(model_dir, model_name)

            if not os.path.exists(model_path):
                print(f"⚠️ Cảnh báo: Không tìm thấy weights model tại {model_path}")
                return True, 1.0

            param = {
                "org_img": img,
                "bbox": bbox,
                "scale": 2.7,
                "out_w": 80,
                "out_h": 80,
                "crop": True
            }

            # Cắt ảnh vùng mặt bằng CropImage
            cropped_img = self.cropper.crop(**param)

            # Dự đoán Liveness
            prediction = self.model_test.predict(cropped_img, model_path)

            label = np.argmax(prediction)
            value = prediction[0][label]

            # label == 1 là Mặt thật (Real Face)
            if label == 1 and value > 0.85:
                return True, float(value)
            else:
                return False, float(value)

        except Exception as e:
            print(f"❌ Lỗi kiểm tra Liveness: {e}")
            return True, 1.0  # Fallback nếu gặp lỗi