# config/settings.py
from pathlib import Path

# ============ 기준 경로 ============
# settings.py 위치(config/) 기준으로 한 단계 위 = 프로젝트 루트(SilverCare/)
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / 'ai_server' / 'models'

# Local AI ingress; Main Server forwards timestamped JPEG samples.
CAMERA_IDS = tuple(f'cam_{index:02d}' for index in range(1, 6))
AI_SERVER_PORT = 9100
GUI_VIDEO_PORT = 9998

# TCP 포트
MAIN_SERVICE_HOST = 'localhost'
MAIN_SERVICE_PORT = 5000
GUI_INFO_PORT = 9999

# ============ 모델 경로 ============
YOLO_POSE_MODEL = str(MODEL_DIR / 'yolov8n-pose.pt')

# Fall Detection
FALL_DETECTOR_MODEL = str(MODEL_DIR / 'fall_lstm.pt')

# Gait Analysis (6개 모델) 자체로딩

# Stretching (모델 불필요)