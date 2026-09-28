"""Shared pose model with independent, conservative per-camera IoU identities."""
import numpy as np
from ultralytics import YOLO
from config.settings import YOLO_POSE_MODEL


class YOLOPoseWrapper:
    def __init__(self, device='cuda'):
        self.model = YOLO(YOLO_POSE_MODEL)
        self.device = device
        self.tracks = {}
        self.next_ids = {}

    def reset(self, camera_id):
        self.tracks.pop(camera_id, None)

    def detect_batch(self, frames):
        """frames: [(camera_id, BGR frame), ...]; returns aligned pose dictionaries."""
        if not frames:
            return []
        results = self.model.predict([frame for _, frame in frames], imgsz=640,
                                     conf=0.5, device=self.device, verbose=False)
        output = []
        for (camera_id, _), result in zip(frames, results):
            previous = self.tracks.get(camera_id, {})
            current = {}
            poses = {}
            if result.boxes is not None and result.keypoints is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                points = result.keypoints.xy.cpu().numpy()
                candidates = []
                for box, keypoints in zip(boxes, points):
                    if keypoints.shape != (17, 2):
                        continue
                    for track_id, old_box in previous.items():
                        left = np.maximum(box[:2], old_box[:2])
                        right = np.minimum(box[2:], old_box[2:])
                        intersection = np.prod(np.maximum(right - left, 0))
                        area_a = np.prod(np.maximum(box[2:] - box[:2], 0))
                        area_b = np.prod(np.maximum(old_box[2:] - old_box[:2], 0))
                        score = intersection / (area_a + area_b - intersection + 1e-9)
                        if score >= 0.3:
                            candidates.append((score, len(poses), track_id))
                    poses[len(poses)] = (box, keypoints)
                matches = {}
                used = set()
                for _, index, track_id in sorted(candidates, reverse=True):
                    if index not in matches and track_id not in used:
                        matches[index] = track_id
                        used.add(track_id)
                mapped = {}
                for index, (box, points) in poses.items():
                    track_id = matches.get(index)
                    if track_id is None:
                        track_id = self.next_ids.get(camera_id, 1)
                        self.next_ids[camera_id] = track_id + 1
                    current[track_id] = box
                    mapped[track_id] = points.astype(np.float32)
                poses = mapped
            self.tracks[camera_id] = current
            output.append(poses)
        return output
