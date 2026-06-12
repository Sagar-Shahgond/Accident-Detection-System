"""
TransitGuard AI - Collision Detection Logic
Step 2: Check if two bounding boxes overlap significantly (potential collision)
"""

def calculate_overlap(box1, box2):
    """
    Calculate IoU (Intersection over Union) between two boxes.
    Each box is (x1, y1, x2, y2).
    Returns a value 0.0 (no overlap) to 1.0 (perfect overlap).
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def find_collisions(detections, overlap_threshold=0.05):
    """
    Given a list of detections (each with 'type' and 'box'),
    find all pairs that overlap above the threshold.
    """
    collisions = []

    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            obj1 = detections[i]
            obj2 = detections[j]

            overlap = calculate_overlap(obj1["box"], obj2["box"])

            if overlap > overlap_threshold:
                collisions.append({
                    "obj1": obj1["type"],
                    "obj2": obj2["type"],
                    "overlap": round(overlap, 3),
                    "box1": obj1["box"],
                    "box2": obj2["box"]
                })

    return collisions