"""
TransitGuard AI - Accident Event Detector
Step 3: Track objects across frames, detect sudden new overlaps
combined with sudden velocity drops -> flag as accident event.
"""

from collision_detection import calculate_overlap, find_collisions


def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def get_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def match_objects(prev_detections, curr_detections, max_distance=60):
    matches = []
    used_prev = set()

    for curr in curr_detections:
        curr_center = get_center(curr["box"])
        best_match = None
        best_dist = max_distance

        for idx, prev in enumerate(prev_detections):
            if idx in used_prev or prev["type"] != curr["type"]:
                continue

            prev_center = get_center(prev["box"])
            dist = get_distance(prev_center, curr_center)

            if dist < best_dist:
                best_dist = dist
                best_match = idx

        if best_match is not None:
            used_prev.add(best_match)
            matches.append((prev_detections[best_match], curr, best_dist))

    return matches


class AccidentDetector:
    """
    Maintains state across frames to detect accident events.

    KEY CHANGES from v1:
    - overlap_threshold raised to 0.15 (was 0.05) - ignores minor/grazing overlaps
    - person<->person pairs ignored entirely (not vehicle accidents)
    - event ONLY fires if overlap is NEW *and* a sudden stop is detected
      (was firing on new overlap alone, causing many false positives)
    """

    def __init__(self, overlap_threshold=0.15, sudden_stop_speed=3.0, history_size=5):
        self.overlap_threshold = overlap_threshold
        self.sudden_stop_speed = sudden_stop_speed
        self.history_size = history_size

        self.prev_detections = []
        self.prev_pairs_overlapping = set()
        self.speed_history = {}

    def _pair_key(self, obj1, obj2):
        c1 = get_center(obj1["box"])
        c2 = get_center(obj2["box"])
        return (obj1["type"], round(c1[0] / 50), round(c1[1] / 50),
                obj2["type"], round(c2[0] / 50), round(c2[1] / 50))

    def process_frame(self, frame_num, detections):
        events = []

        matches = match_objects(self.prev_detections, detections)
        speeds = {}

        for prev, curr, dist in matches:
            speeds[id(curr)] = dist

        collisions = find_collisions(detections, self.overlap_threshold)

        for c in collisions:
            obj1_data = next(d for d in detections if d["box"] == c["box1"])
            obj2_data = next(d for d in detections if d["box"] == c["box2"])

            # Ignore person<->person overlaps - not a vehicle accident
            if c["obj1"] == "person" and c["obj2"] == "person":
                continue

            pair_key = self._pair_key(obj1_data, obj2_data)
            is_new_overlap = pair_key not in self.prev_pairs_overlapping

            speed1 = speeds.get(id(obj1_data), None)
            speed2 = speeds.get(id(obj2_data), None)

            sudden_stop = False
            for s in (speed1, speed2):
                if s is not None and s < self.sudden_stop_speed:
                    sudden_stop = True

            # Require BOTH: new overlap AND sudden stop
            if is_new_overlap and sudden_stop:
                pair_type = "vehicle-pedestrian" if "person" in (c["obj1"], c["obj2"]) else "vehicle-vehicle"

                events.append({
                    "frame": frame_num,
                    "type": pair_type,
                    "objects": (c["obj1"], c["obj2"]),
                    "overlap": c["overlap"],
                    "sudden_stop": sudden_stop,
                    "speeds": (speed1, speed2),
                })

        self.prev_pairs_overlapping = set()
        for c in collisions:
            if c["obj1"] == "person" and c["obj2"] == "person":
                continue
            obj1_data = next(d for d in detections if d["box"] == c["box1"])
            obj2_data = next(d for d in detections if d["box"] == c["box2"])
            self.prev_pairs_overlapping.add(self._pair_key(obj1_data, obj2_data))

        self.prev_detections = detections

        return events