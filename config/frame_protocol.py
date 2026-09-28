"""Binary WebSocket frame: 4-byte big-endian JSON header length, JSON header, JPEG."""
import json

MAX_JPEG_BYTES = 2_000_000
MAX_HEADER_BYTES = 4096


def pack_frame(metadata, jpeg):
    header = json.dumps(metadata, separators=(',', ':')).encode('utf-8')
    if not 0 < len(header) <= MAX_HEADER_BYTES or not 4 <= len(jpeg) <= MAX_JPEG_BYTES:
        raise ValueError('invalid frame size')
    return len(header).to_bytes(4, 'big') + header + jpeg


def unpack_frame(message):
    if not isinstance(message, bytes) or len(message) < 8:
        raise ValueError('binary frame required')
    size = int.from_bytes(message[:4], 'big')
    if not 0 < size <= MAX_HEADER_BYTES or len(message) - 4 - size > MAX_JPEG_BYTES:
        raise ValueError('invalid frame size')
    metadata = json.loads(message[4:4 + size])
    jpeg = message[4 + size:]
    if not isinstance(metadata, dict) or jpeg[:2] != b'\xff\xd8' or jpeg[-2:] != b'\xff\xd9':
        raise ValueError('invalid JPEG frame')
    return metadata, jpeg
