"""WebSocket video ingress with the unchanged main-version model outputs."""
import json
import logging
import time

from config.settings import CAMERA_IDS
from .ai_manager import AIManager
from .ai_networkmanager import AINetworkManager

LOG = logging.getLogger(__name__)


class AIService:
    def __init__(self, device='cuda'):
        self.device = device
        self.net = AINetworkManager()
        self.running = False

    def run(self):
        managers = {camera_id: AIManager(camera_id, self.device) for camera_id in CAMERA_IDS}
        self.net.start()
        self.running = True
        try:
            while self.running:
                started = time.monotonic()
                for camera_id, (frame_id, capture_time, mode, frame) in self.net.take_latest().items():
                    if time.time() - capture_time > 0.5:
                        managers[camera_id] = AIManager(camera_id, self.device)
                        continue
                    try:
                        manager = managers[camera_id]
                        if manager.current_mode != mode:
                            managers[camera_id] = manager = AIManager(camera_id, self.device)
                        manager.set_mode(mode)
                        # Main-version AIManager and RawDetection retain their outputs.
                        result = json.loads(manager.process_frame(frame))
                        pose = manager.yolo_pose.last_result
                        # Transport metadata only: retain frame_idx, detections, raw_data as-is.
                        result['frame_id'] = frame_id
                        result['skeletons'] = [
                            {'track_id': int(track_id), 'keypoints': points.tolist()}
                            for track_id, points in pose['keypoints'].items()
                        ]
                        self.net.send_to_main(json.dumps(result, ensure_ascii=False))
                    except Exception:
                        LOG.exception('Inference failed for %s', camera_id)
                        managers[camera_id] = AIManager(camera_id, self.device)
                time.sleep(max(0, 1 / 15 - (time.monotonic() - started)))
        finally:
            self.net.stop()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    AIService().run()
