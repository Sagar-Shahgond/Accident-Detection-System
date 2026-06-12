"""
TransitGuard AI - Severity Scoring
Step 4: Convert an accident event (from accident_detector.py) into a
severity score (0-100) and level (MINOR / MAJOR / CRITICAL).
"""


def calculate_severity_score(event: dict) -> dict:
    """
    Takes an accident event dict (as produced by AccidentDetector.process_frame)
    and returns a severity score + level.
    """
    score = 0
    breakdown = {}

    # 1. Pedestrian involvement (highest weight)
    if event["type"] == "vehicle-pedestrian":
        score += 35
        breakdown["pedestrian_involved"] = 35
    else:
        breakdown["pedestrian_involved"] = 0

    # 2. Overlap severity
    overlap = event.get("overlap", 0)
    overlap_points = min(int(overlap * 50), 25)
    score += overlap_points
    breakdown["overlap_points"] = overlap_points

    # 3. Sudden stop
    if event.get("sudden_stop"):
        score += 25
        breakdown["sudden_stop"] = 25
    else:
        breakdown["sudden_stop"] = 0

    # 4. Speed before impact
    speeds = event.get("speeds", (None, None))
    valid_speeds = [s for s in speeds if s is not None]
    max_speed = max(valid_speeds) if valid_speeds else 0

    if max_speed > 15:
        speed_points = 15
    elif max_speed > 8:
        speed_points = 10
    elif max_speed > 3:
        speed_points = 5
    else:
        speed_points = 0

    score += speed_points
    breakdown["speed_points"] = speed_points
    breakdown["max_speed_observed"] = round(max_speed, 2)

    # 5. Large vehicle bonus
    if "truck" in event["objects"] or "bus" in event["objects"]:
        score += 5
        breakdown["large_vehicle_bonus"] = 5
    else:
        breakdown["large_vehicle_bonus"] = 0

    score = min(score, 100)

    if score <= 40:
        level = "MINOR"
    elif score <= 70:
        level = "MAJOR"
    else:
        level = "CRITICAL"

    return {
        "score": score,
        "level": level,
        "breakdown": breakdown
    }


if __name__ == "__main__":
    print("TEST 1: Real event from your pipeline (Frame 48, car<->car)")
    real_event = {
        "frame": 48,
        "type": "vehicle-vehicle",
        "objects": ("car", "car"),
        "overlap": 0.438,
        "sudden_stop": True,
        "speeds": (5.0, 2.0)
    }
    result1 = calculate_severity_score(real_event)
    print(f"  Score: {result1['score']} -> {result1['level']}")
    print(f"  Breakdown: {result1['breakdown']}\n")

    print("TEST 2: Minor fender bender (low overlap, slow speeds)")
    minor_event = {
        "frame": 10,
        "type": "vehicle-vehicle",
        "objects": ("car", "car"),
        "overlap": 0.08,
        "sudden_stop": True,
        "speeds": (2.0, 1.0)
    }
    result2 = calculate_severity_score(minor_event)
    print(f"  Score: {result2['score']} -> {result2['level']}")
    print(f"  Breakdown: {result2['breakdown']}\n")

    print("TEST 3: Critical (pedestrian, high speed, high overlap)")
    critical_event = {
        "frame": 99,
        "type": "vehicle-pedestrian",
        "objects": ("car", "person"),
        "overlap": 0.6,
        "sudden_stop": True,
        "speeds": (20.0, 0.5)
    }
    result3 = calculate_severity_score(critical_event)
    print(f"  Score: {result3['score']} -> {result3['level']}")
    print(f"  Breakdown: {result3['breakdown']}\n")

    print("TEST 4: Major (truck involved, moderate everything)")
    major_event = {
        "frame": 75,
        "type": "vehicle-vehicle",
        "objects": ("car", "truck"),
        "overlap": 0.25,
        "sudden_stop": True,
        "speeds": (10.0, 1.0)
    }
    result4 = calculate_severity_score(major_event)
    print(f"  Score: {result4['score']} -> {result4['level']}")
    print(f"  Breakdown: {result4['breakdown']}")