# config/settings.py
from pathlib import Path

# ============ 기준 경로 ============
# settings.py 위치(config/) 기준으로 한 단계 위 = 프로젝트 루트(SilverCare/)
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / 'ai_server' / 'models'

# WebSocket JPEG ingress on loopback; HTTP result callback on loopback.
CAMERA_IDS = tuple(f'cam_{index:02d}' for index in range(1, 6))

AI_SERVER_PORT = 9100
MAIN_SERVICE_HOST = '127.0.0.1'
MAIN_SERVICE_PORT = 5000

# ============ 모델 경로 ============
YOLO_POSE_MODEL = str(MODEL_DIR / 'yolov8n-pose.pt')

# Fall Detection
FALL_DETECTOR_MODEL = str(MODEL_DIR / 'fall_lstm.pt')

# Gait Analysis (6개 모델) 자체로딩

# Stretching (모델 불필요)
