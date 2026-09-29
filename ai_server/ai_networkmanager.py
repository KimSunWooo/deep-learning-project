"""Local JPEG input WebSocket and AIOutput HTTP callback."""
import asyncio
import json
import logging
import threading
import time
from urllib.request import Request, urlopen
from urllib.error import URLError

import cv2
import numpy as np
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from config.frame_protocol import unpack_frame, MAX_JPEG_BYTES, MAX_HEADER_BYTES
from config.settings import AI_SERVER_PORT, MAIN_SERVICE_HOST, MAIN_SERVICE_PORT, CAMERA_IDS


class AINetworkManager:
    def __init__(self, port=AI_SERVER_PORT, main_host=MAIN_SERVICE_HOST, main_port=MAIN_SERVICE_PORT):
        self.port = port
        self.main_host = main_host
        self.main_port = main_port
        self.frames = {}
        self.last_ids = {}
        self.lock = threading.Lock()
        self.frame_ready = threading.Condition(self.lock)
        self.thread = None
        self.loop = None
        self.socket = None
        self.ready = threading.Event()
        self.stop_requested = threading.Event()
        self.logger = logging.getLogger(__name__)

    async def _handler(self, socket):
        if socket.request.path != '/frames':
            await socket.close(code=1008, reason='unknown endpoint')
            return
        if self.socket is not None:
            await socket.close(code=1008, reason='main server already connected')
            return
        self.socket = socket
        # A restarted main server begins a new frame-id sequence.
        with self.frame_ready:
            self.last_ids.clear()
            self.frames.clear()
        try:
            async for message in socket:
                try:
                    metadata, jpeg = unpack_frame(message)
                    camera_id = metadata['camera_id']
                    frame_id = metadata['frame_id']
                    timestamp = metadata['capture_time']
                    mode = metadata['mode']
                    if (camera_id not in CAMERA_IDS or type(frame_id) is not int or
                            type(timestamp) not in (int, float) or type(mode) is not int or
                            mode not in (0, 1, 2) or abs(time.time() - timestamp) > 2):
                        raise ValueError('invalid metadata or stale frame')
                    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError('invalid JPEG')
                    with self.frame_ready:
                        if frame_id <= self.last_ids.get(camera_id, -1):
                            continue
                        self.last_ids[camera_id] = frame_id
                        self.frames[camera_id] = (frame_id, timestamp, mode, image)
                        self.frame_ready.notify_all()
                except (ValueError, KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError):
                    self.logger.warning('Dropped invalid AI input frame')
        except ConnectionClosed:
            pass
        finally:
            if self.socket is socket:
                self.socket = None

    def start(self):
        async def run():
            self.loop = asyncio.get_running_loop()
            async with serve(self._handler, '127.0.0.1', self.port,
                             max_size=MAX_JPEG_BYTES + MAX_HEADER_BYTES + 4,
                             compression=None):
                self.ready.set()
                while not self.stop_requested.is_set():
                    await asyncio.sleep(0.1)

        def target():
            try:
                asyncio.run(run())
            except Exception:
                self.logger.exception('AI video WebSocket server failed')
                self.ready.set()

        self.thread = threading.Thread(target=target, daemon=True)
        self.thread.start()
        if not self.ready.wait(timeout=5) or self.loop is None:
            raise RuntimeError('AI video WebSocket server did not start')

    def take_latest(self):
        with self.lock:
            frames = self.frames
            self.frames = {}
        return frames

    def get_frame(self, camera_id):
        """Wait for this camera's newest frame; keep other cameras independent."""
        with self.frame_ready:
            while camera_id not in self.frames:
                self.frame_ready.wait()
            frame_id, capture_time, mode, frame = self.frames.pop(camera_id)
        return {'frame_id': frame_id, 'capture_time': capture_time,
                'mode': mode, 'frame': frame}

    def send_to_main(self, result):
        try:
            request = Request(f'http://{self.main_host}:{self.main_port}/ai_result',
                              data=result.encode('utf-8'),
                              headers={'Content-Type': 'application/json'}, method='POST')
            with urlopen(request, timeout=1) as response:
                if response.status != 200:
                    raise URLError(f'Main returned {response.status}')
        except (URLError, TimeoutError):
            self.logger.exception('AI result delivery failed')

    def stop(self):
        self.stop_requested.set()
        if self.thread:
            self.thread.join(timeout=1)
