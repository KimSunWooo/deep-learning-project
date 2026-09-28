"""Example camera agent: webcam JPEG @30 FPS over one WebSocket."""
import argparse
import asyncio
import time
import cv2
from websockets.asyncio.client import connect


async def stream(url, source):
    capture = cv2.VideoCapture(source)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    capture.set(cv2.CAP_PROP_FPS, 30)
    if not capture.isOpened():
        raise RuntimeError(f'Cannot open camera {source}')
    try:
        while True:
            try:
                async with connect(url, max_size=2_000_000) as socket:
                    while True:
                        started = time.monotonic()
                        ok, frame = await asyncio.to_thread(capture.read)
                        if not ok:
                            raise RuntimeError('Camera read failed')
                        ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        if ok:
                            await socket.send(jpeg.tobytes())
                        await asyncio.sleep(max(0, 1 / 30 - (time.monotonic() - started)))
            except (OSError, ConnectionError) as exc:
                print(f'Connection lost: {exc}; retrying')
                await asyncio.sleep(2)
    finally:
        capture.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='ws://127.0.0.1:8765/cameras/cam_01?mode=0')
    parser.add_argument('--source', default='0', help='Webcam index or video URL')
    arguments = parser.parse_args()
    source = int(arguments.source) if arguments.source.isdigit() else arguments.source
    asyncio.run(stream(arguments.url, source))
