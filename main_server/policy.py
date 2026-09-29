"""Connect main-version decision modules without altering AI model output or policy code."""
import copy
import json
import os
from pathlib import Path
import numpy as np

from main_service.fall_analyzer import FallAnalyzer
from main_service.stretching_analyzer import StretchingAnalyzer

LIMBS = ('left_arm', 'right_arm', 'left_leg', 'right_leg')
SPREADS = ('arm_spread', 'leg_spread')


class DecisionEngine:
    def __init__(self, reference_path=None):
        self.fall = FallAnalyzer()
        reference_path = reference_path or os.getenv('STRETCH_REFERENCE_PATH')
        self.stretch = None
        if reference_path:
            data = json.loads(Path(reference_path).read_text(encoding='utf-8'))
            # Reference poses are supplied in source-image pixel coordinates;
            # main StretchingManager reports shoulder-centered keypoints.
            data = copy.deepcopy(data)
            for skeleton in data['skeletons']:
                points = np.asarray(skeleton['keypoints'], dtype=np.float32)
                if points.shape != (17, 2):
                    raise ValueError('reference must contain COCO-17 coordinates')
                center = (points[5] + points[6]) / 2
                skeleton['keypoints'] = (points - center).tolist()
            self.stretch = StretchingAnalyzer(data)

    def evaluate(self, result):
        mode = result['mode']
        if mode == 0:
            return {'fall': self.fall.analyze(result)}
        if mode == 1:
            people = []
            for detection in result['detections']:
                values = detection['raw_data']
                # Prefer main-version cumulative average; the raw scores remain untouched.
                scores = values.get('cumulative_scores', values['disease_scores'])
                category = max(scores, key=scores.get) if scores else None
                people.append({'track_id': detection['track_id'], 'category': category,
                               'scores_used': 'cumulative_scores' if 'cumulative_scores' in values else 'disease_scores'})
            return {'gait': {'people': people, 'status': 'ready' if people else 'insufficient_frames'}}
        if mode == 2:
            if self.stretch is None:
                return {'stretch': {'status': 'reference_missing'}}
            # main-version StretchingManager emits lists; main-version
            # StretchingAnalyzer indexes named limbs. Adapt a private view only.
            view = copy.deepcopy(result)
            for detection in view['detections']:
                raw = detection['raw_data']
                angles, spreads = raw['limb_angles'], raw['spread_distances']
                if len(angles) != len(LIMBS) or len(spreads) != len(SPREADS):
                    raise ValueError('invalid stretching feature count')
                raw['limb_angles'] = dict(zip(LIMBS, angles))
                raw['spread_distances'] = dict(zip(SPREADS, spreads))
            return {'stretch': self.stretch.analyze(view)}
        raise ValueError('unknown mode')
