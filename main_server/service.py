"""Camera and AI input WebSockets, local AI result HTTP, server-rendered UI viewing."""
import asyncio
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import os
import threading
import time
import cv2
import numpy as np
from urllib.parse import urlparse

from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from config.settings import CAMERA_IDS
from config.frame_protocol import pack_frame
from main_server.policy import DecisionEngine

LOG = logging.getLogger(__name__)
AI_URL = os.getenv('AI_VIDEO_URL', 'ws://127.0.0.1:9100/frames')
MAX_FRAME_BYTES = 2_000_000
SAMPLE_INTERVAL = 1 / 15
SKELETON_EDGES = ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                  (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
                  (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4))


def render_skeletons(jpeg, skeletons):
    """Draw COCO-17 pixel coordinates over the exact frame that AI analyzed."""
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('invalid source JPEG')
    height, width = image.shape[:2]
    if not isinstance(skeletons, list) or len(skeletons) > 50:
        raise ValueError('invalid skeleton count')
    for skeleton in skeletons:
        points = skeleton['keypoints']
        if not isinstance(points, list) or len(points) != 17:
            raise ValueError('expected COCO-17 keypoints')
        valid = []
        for point in points:
            if not isinstance(point, list) or len(point) != 2 or any(type(v) not in (int, float) or not np.isfinite(v) for v in point):
                raise ValueError('invalid keypoint')
            x, y = point
            valid.append((int(x), int(y)) if 0 < x < width and 0 < y < height else None)
        for a, b in SKELETON_EDGES:
            if valid[a] is not None and valid[b] is not None:
                cv2.line(image, valid[a], valid[b], (0, 255, 0), 2)
        for point in valid:
            if point is not None:
                cv2.circle(image, point, 3, (0, 255, 0), -1)
        first = next((point for point in valid if point is not None), None)
        if first is not None:
            cv2.putText(image, f"ID {skeleton['track_id']}",
                        (first[0], max(15, first[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
    ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise ValueError('JPEG encoding failed')
    return encoded.tobytes()


DASHBOARD_HTML = """<!doctype html>
<html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dolbom 모니터링</title><style>
body{font:16px system-ui,sans-serif;max-width:1280px;margin:auto;padding:20px;background:#f4f6fa;color:#172230}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}
article{background:white;border-radius:12px;padding:14px;box-shadow:0 2px 12px #0001}
img{width:100%;aspect-ratio:16/9;object-fit:contain;background:#111}
pre{white-space:pre-wrap;word-break:break-word;max-height:220px;overflow:auto}
</style><h1>Dolbom 카메라 모니터링</h1><main id="cameras"></main>
<script>
const root=document.getElementById('cameras');
for(let i=1;i<=5;i++){
 const id='cam_'+String(i).padStart(2,'0');
 const card=document.createElement('article');
 const title=document.createElement('h2');title.textContent=id;
 const status=document.createElement('p');status.id=id+'-status';
 const image=document.createElement('img');image.src='/preview/'+id;image.alt=id+' 영상';
 const result=document.createElement('pre');result.id=id+'-result';
 card.append(title,status,image,result);root.append(card);
}
async function refresh(){
 try{
  const health=await (await fetch('/health',{cache:'no-store'})).json();
  for(const [id,state] of Object.entries(health)){
   document.getElementById(id+'-status').textContent=(state.connected?'연결됨':'연결 끊김')+' · 프레임 '+state.last_frame_id+' · AI 결과 '+state.last_result_id;
   const result=await (await fetch('/results/'+id,{cache:'no-store'})).json();
   document.getElementById(id+'-result').textContent=JSON.stringify(result,null,2);
  }
 }catch(error){console.error(error)}
}
refresh();setInterval(refresh,1000);
</script></html>"""


@dataclass
class CameraState:
    mode: int = 0
    connection: object = None
    jpeg: bytes = b''
    frame_id: int = 0
    captured_at: float = 0.0
    received_at: float = 0.0
    result: dict = field(default_factory=dict)
    last_result_id: int = -1
    next_sample: float = 0.0
    last_sent_id: int = 0
    forwarding: bool = False
    processed_jpeg: bytes = b''
    processed_frame_id: int = -1
    pending_frames: dict = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)


class MainServer:
    def __init__(self, websocket_host='0.0.0.0', websocket_port=8765,
                 http_host='127.0.0.1', http_port=5000, ai_url=AI_URL,
                 ui_host='0.0.0.0', ui_port=5001):
        self.websocket_host = websocket_host
        self.websocket_port = websocket_port
        self.http_host = http_host
        self.http_port = http_port
        self.ai_url = ai_url
        self.ui_host = ui_host
        self.ui_port = ui_port
        self.cameras = {name: CameraState() for name in CAMERA_IDS}
        self.ai_socket = None
        self.loop = None
        self.http_server = None
        self.ui_server = None
        self.ui_clients = set()
        self.decisions = DecisionEngine()
        self.decision_lock = threading.Lock()

    async def ui_handler(self, socket):
        if urlparse(socket.request.path).path != '/ui':
            await socket.close(code=1008, reason='unknown endpoint')
            return
        self.ui_clients.add(socket)
        try:
            async for _ in socket:
                await socket.close(code=1008, reason='UI is receive only')
        finally:
            self.ui_clients.discard(socket)

    async def publish_result(self, result):
        if not self.ui_clients:
            return
        await self.broadcast(result)
        fall = result.get('judgment', {}).get('fall')
        if fall:
            for event in fall['events']:
                if event['event'].endswith('_TO_ALERT'):
                    await self.broadcast({
                        'type': 'event',
                        'event_id': f"{result['camera_id']}:{event['track_id']}:{result['frame_id']}",
                        'camera_id': result['camera_id'], 'severity': 'urgent',
                        'navigate_to': 'cctv', 'content': '낙상 경보: 상태 판단 결과 alert',
                        'occurred_at': event['timestamp'],
                    })

    async def broadcast(self, payload):
        message = json.dumps(payload)
        for client in tuple(self.ui_clients):
            try:
                await asyncio.wait_for(client.send(message), timeout=0.2)
            except Exception:
                self.ui_clients.discard(client)

    async def camera_handler(self, socket):
        parsed = urlparse(socket.request.path)
        parts = parsed.path.strip('/').split('/')
        if len(parts) != 2 or parts[0] != 'cameras' or parts[1] not in self.cameras:
            await socket.close(code=1008, reason='unknown camera')
            return
        camera_id = parts[1]
        from urllib.parse import parse_qs
        try:
            mode = int(parse_qs(parsed.query).get('mode', ['0'])[0])
        except ValueError:
            mode = -1
        if mode not in (0, 1, 2):
            await socket.close(code=1008, reason='invalid mode')
            return
        state = self.cameras[camera_id]
        with state.lock:
            occupied = state.connection is not None
            if not occupied:
                state.connection = socket
        if occupied:
            await socket.close(code=1008, reason='camera already connected')
            return
        with state.lock:
            state.mode = mode
            state.jpeg = b''
            state.last_sent_id = state.frame_id
            state.next_sample = 0
            state.result = {}
            state.processed_jpeg = b''
            state.processed_frame_id = -1
            state.pending_frames.clear()
        try:
            async for message in socket:
                if isinstance(message, str):
                    await socket.close(code=1008, reason='camera sends JPEG binary only')
                    break
                if len(message) > MAX_FRAME_BYTES or len(message) < 4 or message[:2] != b'\xff\xd8' or message[-2:] != b'\xff\xd9':
                    await socket.close(code=1009, reason='invalid JPEG or size')
                    break
                with state.lock:
                    state.frame_id += 1
                    state.jpeg = bytes(message)
                    state.captured_at = time.time()
                    state.received_at = time.monotonic()
        except ConnectionClosed:
            pass
        finally:
            with state.lock:
                if state.connection is socket:
                    state.connection = None
                    state.jpeg = b''

    async def ai_connection(self):
        while True:
            try:
                async with connect(self.ai_url, max_size=MAX_FRAME_BYTES + 4096,
                                   compression=None) as socket:
                    self.ai_socket = socket
                    async for _ in socket:
                        await socket.close(code=1008, reason='AI returns JSON by HTTP')
            except (OSError, ConnectionClosed) as exc:
                LOG.warning('AI video connection unavailable: %s', exc)
            finally:
                self.ai_socket = None
            await asyncio.sleep(1)

    async def scheduler(self):
        while True:
            started = time.monotonic()
            for camera_id, state in self.cameras.items():
                with state.lock:
                    if state.connection is None or not state.jpeg or started < state.next_sample:
                        continue
                    state.next_sample = started + SAMPLE_INTERVAL
                    if started - state.received_at > 0.25 or state.frame_id == state.last_sent_id:
                        continue
                    state.last_sent_id = state.frame_id
                    frame_id = state.frame_id
                    source_jpeg = state.jpeg
                    payload = pack_frame({'camera_id': camera_id, 'frame_id': frame_id,
                                          'capture_time': state.captured_at, 'mode': state.mode}, source_jpeg)
                if self.ai_socket is not None:
                    with state.lock:
                        state.pending_frames[frame_id] = source_jpeg
                        while len(state.pending_frames) > 12:
                            state.pending_frames.pop(next(iter(state.pending_frames)))
                    try:
                        await asyncio.wait_for(self.ai_socket.send(payload), timeout=0.2)
                    except (ConnectionClosed, TimeoutError):
                        with state.lock:
                            state.pending_frames.pop(frame_id, None)
                        LOG.warning('AI frame send failed for %s', camera_id)
            await asyncio.sleep(max(0, SAMPLE_INTERVAL - (time.monotonic() - started)))

    def make_http_handler(self, internal=False):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def send_json(self, status, payload):
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                if not internal or self.path != '/ai_result':
                    self.send_error(404)
                    return
                try:
                    size = int(self.headers.get('Content-Length', '0'))
                    if not 0 < size <= 100_000:
                        raise ValueError('result too large')
                    result = json.loads(self.rfile.read(size))
                    camera_id = result['camera_id']
                    frame_id = result['frame_id']
                    if camera_id not in owner.cameras or not isinstance(frame_id, int):
                        raise ValueError('invalid result identity')
                    state = owner.cameras[camera_id]
                    with state.lock:
                        if frame_id <= state.last_result_id or frame_id > state.frame_id:
                            raise ValueError('stale result')
                        original = state.pending_frames.pop(frame_id, None)
                        if original is None:
                            raise ValueError('source frame expired or not sent')
                        for older in tuple(state.pending_frames):
                            if older < frame_id:
                                del state.pending_frames[older]
                    rendered = render_skeletons(original, result['skeletons'])
                    with owner.decision_lock:
                        # AIOutput remains intact; final state is a separate main-server field.
                        result['judgment'] = owner.decisions.evaluate(result)
                    with state.lock:
                        if frame_id <= state.last_result_id:
                            raise ValueError('stale result')
                        state.last_result_id = frame_id
                        state.result = result
                        state.processed_frame_id = frame_id
                        state.processed_jpeg = rendered
                    asyncio.run_coroutine_threadsafe(owner.publish_result(result), owner.loop).result(timeout=1)
                    self.send_json(200, {'ok': True})
                except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                    self.send_error(400)
                except Exception:
                    LOG.exception('Result delivery failed')
                    self.send_error(503)

            def do_GET(self):
                parsed = urlparse(self.path)
                if parsed.path == '/':
                    if internal:
                        self.send_error(404)
                        return
                    body = DASHBOARD_HTML.encode('utf-8')
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if parsed.path == '/health':
                    status = {}
                    for camera_id, state in owner.cameras.items():
                        with state.lock:
                            status[camera_id] = {'connected': state.connection is not None,
                                                 'last_frame_id': state.frame_id,
                                                 'last_result_id': state.last_result_id}
                    self.send_json(200, status)
                    return
                if parsed.path.startswith('/results/'):
                    camera_id = parsed.path.removeprefix('/results/')
                    if camera_id not in owner.cameras:
                        self.send_error(404)
                        return
                    state = owner.cameras[camera_id]
                    with state.lock:
                        result = dict(state.result)
                    self.send_json(200, result)
                    return
                if not internal and parsed.path.startswith('/preview/'):
                    camera_id = parsed.path.removeprefix('/preview/')
                    if camera_id not in owner.cameras:
                        self.send_error(404)
                        return
                    self.send_response(200)
                    self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                    self.end_headers()
                    state = owner.cameras[camera_id]
                    last_id = 0
                    try:
                        while True:
                            with state.lock:
                                frame_id, jpeg = state.processed_frame_id, state.processed_jpeg
                            if jpeg and frame_id != last_id:
                                self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(jpeg)).encode() + b'\r\n\r\n' + jpeg + b'\r\n')
                                self.wfile.flush()
                                last_id = frame_id
                            time.sleep(1 / 30)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                self.send_error(404)

            def log_message(self, format, *args):
                LOG.debug(format, *args)

        return Handler

    async def run(self):
        self.loop = asyncio.get_running_loop()
        self.http_server = ThreadingHTTPServer((self.http_host, self.http_port), self.make_http_handler(internal=True))
        self.ui_server = ThreadingHTTPServer((self.ui_host, self.ui_port), self.make_http_handler())
        threading.Thread(target=self.http_server.serve_forever, daemon=True).start()
        threading.Thread(target=self.ui_server.serve_forever, daemon=True).start()
        LOG.info('AI callback %s:%s; UI %s:%s; camera WebSocket %s:%s; AI video %s',
                 self.http_host, self.http_port, self.ui_host, self.ui_port,
                 self.websocket_host, self.websocket_port, self.ai_url)
        receiver = asyncio.create_task(self.ai_connection())
        try:
            async with serve(self.camera_handler, self.websocket_host, self.websocket_port,
                             max_size=MAX_FRAME_BYTES, compression=None):
                async with serve(self.ui_handler, self.ui_host, self.ui_port + 1,
                                 max_size=100_000, compression=None):
                    await self.scheduler()
        finally:
            self.http_server.shutdown()
            self.http_server.server_close()
            self.ui_server.shutdown()
            self.ui_server.server_close()
            receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    asyncio.run(MainServer().run())
