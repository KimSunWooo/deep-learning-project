"""One GPU inference owner; camera-specific temporal state."""
import logging
import time
import json
from .ai_networkmanager import AINetworkManager
from .ai_manager import AIManager
from .yolo_pose import YOLOPoseWrapper
from config.settings import CAMERA_IDS


class AIService:
    def __init__(self, device='cuda'):
        self.net = AINetworkManager()
        self.device = device
        self.logger = logging.getLogger(__name__)
        self.running = False

    def run(self):
        managers = {camera_id: AIManager(camera_id, self.device) for camera_id in CAMERA_IDS}
        pose = YOLOPoseWrapper(self.device)
        self.net.start()
        self.running = True
        try:
            while self.running:
                start = time.monotonic()
                items = self.net.take_latest()
                fresh = []
                for camera_id, payload in items.items():
                    frame_id, capture_time, mode, frame = payload
                    if time.time() - capture_time > 0.5:
                        pose.reset(camera_id)
                        managers[camera_id] = AIManager(camera_id, self.device)
                        continue
                    fresh.append((camera_id, frame_id, mode, frame))
                if fresh:
                    try:
                        poses = pose.detect_batch([(cam, frame) for cam, _, _, frame in fresh])
                        for (camera_id, frame_id, mode, frame), keypoints in zip(fresh, poses):
                            manager = managers[camera_id]
                            if manager.current_mode != mode:
                                managers[camera_id] = manager = AIManager(camera_id, self.device)
                                pose.reset(camera_id)
                            manager.set_mode(mode)
                            result = json.loads(manager.process_frame(frame, keypoints, frame_id))
                            result['skeletons'] = [
                                {'track_id': int(track_id), 'keypoints': points.tolist()}
                                for track_id, points in keypoints.items()
                            ]
                            self.net.send_to_main(json.dumps(result, ensure_ascii=False))
                    except Exception:
                        self.logger.exception('Inference batch failed; resetting temporal state')
                        for camera_id, *_ in fresh:
                            pose.reset(camera_id)
                            managers[camera_id] = AIManager(camera_id, self.device)
                time.sleep(max(0, 1 / 15 - (time.monotonic() - start)))
        finally:
            self.net.stop()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    AIService().run()
