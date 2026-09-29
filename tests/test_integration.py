"""Contract tests without loading or running the team's GPU models."""
import asyncio
import copy
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from urllib.request import urlopen

import cv2
import numpy as np
from websockets.asyncio.client import connect

from ai_server.ai_networkmanager import AINetworkManager
from main_server.policy import DecisionEngine
from main_server.service import MainServer


class DecisionContractTest(unittest.TestCase):
    def test_fall_uses_main_version_policy_and_preserves_raw_score(self):
        engine = DecisionEngine()
        result = {'camera_id': 'cam_01', 'frame_idx': 0, 'frame_id': 1,
                  'timestamp': datetime.now(timezone.utc).isoformat(), 'mode': 0,
                  'detections': [{'track_id': 1, 'bbox': [1, 2, 3, 4],
                                  'raw_data': {'fall_prob': 0.8, 'confidence': 1.0}}]}
        raw = copy.deepcopy(result)
        judgment = engine.evaluate(result)
        self.assertEqual(result, raw)
        self.assertEqual(judgment['fall']['camera_status'], 'alert')
        self.assertEqual(judgment['fall']['tracks'][0]['bbox'], [1, 2, 3, 4])

    def test_stretch_adapter_keeps_wire_format_unchanged(self):
        points = [[float(i * 2), float(i * 3)] for i in range(17)]
        with tempfile.TemporaryDirectory() as folder:
            reference = Path(folder) / 'reference.json'
            reference.write_text(json.dumps({'movement_name': 'test', 'skeletons': [
                {'frame_index': 8, 'keypoints': points, 'confidence': [1.0] * 17}]}))
            result = {'camera_id': 'cam_05', 'frame_idx': 8, 'frame_id': 101,
                      'mode': 2, 'detections': [{'track_id': 2, 'bbox': [0, 0, 50, 50],
                      'raw_data': {'keypoints': points, 'limb_angles': [0., 0., 0., 0.],
                                   'spread_distances': [10., 10.], 'confidence': 1.0}}]}
            original = copy.deepcopy(result)
            judgment = DecisionEngine(str(reference)).evaluate(result)
            self.assertEqual(result, original)
            self.assertIn(2, judgment['stretch']['tracking_data'])

    def test_gait_main_scores_remain_unchanged(self):
        result = {'mode': 1, 'detections': [{'track_id': 7, 'raw_data': {
            'disease_scores': {'normal': .2, 'stroke': .8},
            'cumulative_scores': {'normal': .7, 'stroke': .3}}}]}
        original = copy.deepcopy(result)
        judgment = DecisionEngine().evaluate(result)
        self.assertEqual(result, original)
        self.assertEqual(judgment['gait']['people'][0]['category'], 'normal')

    def test_ai_transport_adds_frame_identity_without_changing_main_output(self):
        # Use an isolated model stand-in; no torch or GPU is needed for this contract.
        original = {'camera_id': 'cam_01', 'timestamp': '2026-09-29T02:00:00Z',
                    'frame_idx': 9, 'mode': 1, 'detections': [{'track_id': 3,
                    'bbox': [1, 2, 3, 4], 'raw_data': {
                        'disease_scores': {'normal': .7, 'stroke': .3},
                        'cumulative_scores': {'normal': .6, 'stroke': .4}}}]}
        model_module = types.ModuleType('ai_server.ai_manager')

        class Model:
            def __init__(self, camera_id, device):
                self.current_mode = 0
                self.yolo_pose = types.SimpleNamespace(last_result={
                    'keypoints': {3: np.zeros((17, 2), np.float32)}})

            def set_mode(self, mode):
                self.current_mode = mode

            def process_frame(self, frame):
                return json.dumps(original)

        model_module.AIManager = Model
        with patch.dict(sys.modules, {'ai_server.ai_manager': model_module}):
            spec = importlib.util.spec_from_file_location(
                'ai_server._contract_ai_service',
                str(Path(__file__).resolve().parents[1] / 'ai_server' / 'ai_service.py'))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            service = module.AIService(device='cpu')

            class Network:
                def start(self):
                    pass

                def get_frame(self, camera_id):
                    self.assert_camera = camera_id
                    return {'frame_id': 123, 'mode': 1,
                            'frame': np.zeros((10, 10, 3), np.uint8)}

                def send_to_main(self, payload):
                    self.result = json.loads(payload)
                    raise SystemExit

                def stop(self):
                    pass

            network = Network()
            service.net = network
            service.ai_managers['cam_01'] = Model('cam_01', 'cpu')
            with self.assertRaises(SystemExit):
                service._process_camera('cam_01')
            self.assertEqual(network.assert_camera, 'cam_01')
            for name, value in original.items():
                self.assertEqual(network.result[name], value)
            self.assertEqual(network.result['frame_id'], 123)
            self.assertEqual(len(network.result['skeletons'][0]['keypoints']), 17)


def preview_frame(url):
    with urlopen(url, timeout=3) as response:
        while response.readline().strip() != b'--frame':
            pass
        assert response.readline().strip() == b'Content-Type: image/jpeg'
        length = int(response.readline().strip().split(b':')[1])
        response.readline()
        return cv2.imdecode(np.frombuffer(response.read(length), np.uint8), cv2.IMREAD_COLOR)


class NetworkContractTest(unittest.IsolatedAsyncioTestCase):
    async def test_camera_to_ai_to_policy_to_ui(self):
        ai = AINetworkManager(port=29100, main_port=29500)
        ai.start()
        main = MainServer(websocket_port=28765, http_port=29500, ui_port=29501,
                          ai_url='ws://127.0.0.1:29100/frames')
        task = asyncio.create_task(main.run())
        await asyncio.sleep(.2)
        try:
            async with connect('ws://127.0.0.1:29502/ui') as ui:
                async with connect('ws://127.0.0.1:28765/cameras/cam_01?mode=0') as camera:
                    ok, jpeg = cv2.imencode('.jpg', np.zeros((80, 100, 3), np.uint8))
                    self.assertTrue(ok)
                    await camera.send(jpeg.tobytes())
                    for _ in range(50):
                        items = ai.take_latest()
                        if items:
                            break
                        await asyncio.sleep(.05)
                    self.assertIn('cam_01', items)
                    frame_id, _, mode, _ = items['cam_01']
                    result = {'camera_id': 'cam_01', 'frame_idx': 0,
                              'frame_id': frame_id, 'timestamp': datetime.now(timezone.utc).isoformat(),
                              'mode': mode, 'detections': [{'track_id': 1, 'bbox': [10, 10, 40, 60],
                              'raw_data': {'fall_prob': .8, 'confidence': 1.}}],
                              'skeletons': [{'track_id': 1,
                              'keypoints': [[20, 20]] + [[0, 0]] * 16}]}
                    await asyncio.to_thread(ai.send_to_main, json.dumps(result))
                    received = json.loads(await asyncio.wait_for(ui.recv(), 3))
                    self.assertEqual(received['frame_id'], frame_id)
                    self.assertEqual(received['frame_idx'], 0)
                    self.assertEqual(received['detections'], result['detections'])
                    self.assertEqual(received['judgment']['fall']['camera_status'], 'alert')
                    event = json.loads(await asyncio.wait_for(ui.recv(), 3))
                    self.assertEqual(event['type'], 'event')
                    rendered = await asyncio.to_thread(preview_frame,
                        'http://127.0.0.1:29501/preview/cam_01')
                    self.assertGreater(int(rendered[20, 20, 1]), 100)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            ai.stop()


if __name__ == '__main__':
    unittest.main()
