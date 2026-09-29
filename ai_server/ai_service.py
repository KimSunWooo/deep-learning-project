"""Run the original per-camera AIManager flow over WebSocket/HTTP transport."""
import json
import logging
import threading
import time

from config.settings import CAMERA_IDS
from .ai_manager import AIManager
from .ai_networkmanager import AINetworkManager


class AIService:
    def __init__(self, device='cuda'):
        self.net = AINetworkManager()
        self.ai_managers = {}
        self.device = device
        self.logger = logging.getLogger('AIService')

    def run(self):
        self.net.start()
        self.logger.info('AIService started on WebSocket port 9100')
        for camera_id in CAMERA_IDS:
            ai_mgr = AIManager(camera_id=camera_id, device=self.device)
            self.ai_managers[camera_id] = ai_mgr
            threading.Thread(target=self._process_camera, args=(camera_id,),
                             daemon=True).start()
            self.logger.info('Started thread for %s', camera_id)
        while True:
            threading.Event().wait(1)

    def _process_camera(self, camera_id: str):
        """Preserve main's set_mode -> process_frame -> send result sequence."""
        ai_mgr = self.ai_managers[camera_id]
        while True:
            try:
                payload = self.net.get_frame(camera_id)
                ai_mgr.set_mode(payload['mode'])
                json_result = ai_mgr.process_frame(payload['frame'])

                # Metadata identifies the JPEG held by main; preserve AIOutput fields.
                result = json.loads(json_result)
                result['frame_id'] = payload['frame_id']
                result['skeletons'] = [
                    {'track_id': int(track_id), 'keypoints': points.tolist()}
                    for track_id, points in ai_mgr.yolo_pose.last_result['keypoints'].items()
                ]
                self.net.send_to_main(json.dumps(result, ensure_ascii=False))
            except Exception:
                self.logger.exception('[%s] Error', camera_id)
                time.sleep(0.1)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    AIService().run()
