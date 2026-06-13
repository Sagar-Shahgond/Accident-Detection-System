"""
TransitGuard AI - Computer Vision Pipeline
Step 1+2+3: Read video frame-by-frame, run YOLOv8, track objects,
and detect accident events (sudden overlap + sudden stop)
"""
import requests
from severity import calculate_severity_score
import cv2
from ultralytics import YOLO
from accident_detector import AccidentDetector
from dispatcher_agent import dispatch_incident

# Load pre-trained YOLOv8 model (downloads automatically first time, ~6MB)
model = YOLO("yolov8n.pt")  # 'n' = nano, fastest version, good for hackathon

# Object classes we care about (from COCO dataset class IDs)
# Includes vehicles AND person, since accidents can be vehicle-vehicle
# or vehicle-pedestrian
VEHICLE_CLASSES = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: could not open video {video_path}")
        return

    frame_num = 0
    detector = AccidentDetector()  # NEW: tracks objects across frames
    all_events = []  # NEW: collect every accident event found

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # video ended

        frame_num += 1

        # Run YOLO on this frame
        results = model(frame, verbose=False)[0]

        vehicles_in_frame = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id in VEHICLE_CLASSES:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                vehicles_in_frame.append({
                    "type": VEHICLE_CLASSES[cls_id],
                    "box": (x1, y1, x2, y2),
                    "conf": round(conf, 2)
                })

        if vehicles_in_frame:
            print(f"Frame {frame_num}: {len(vehicles_in_frame)} vehicles -> {vehicles_in_frame}")

        # NEW: feed this frame's detections into the accident detector + severity calculation + printing
        events = detector.process_frame(frame_num, vehicles_in_frame)
        for e in events:
            severity = calculate_severity_score(e)
            e["severity"] = severity  # attach severity to the event
            dispatch = dispatch_incident(
                accident_type=e["type"],
                severity_score=severity["score"],
                severity_level=severity["level"],
                timestamp=f"frame {e['frame']}",
                location="Camera-01 / Main Road Junction",
            )
            e["dispatch"] = dispatch.model_dump()

            print(f"  >>> ACCIDENT EVENT at frame {e['frame']}: "
                  f"{e['objects'][0]} <-> {e['objects'][1]} "
                  f"(overlap={e['overlap']}, sudden_stop={e['sudden_stop']})")
            print(f"      SEVERITY: {severity['score']}/100 -> {severity['level']}")
            print(f"      Breakdown: {severity['breakdown']}")
            print(f"      DISPATCH: {e['dispatch']}")

            # Send this incident to the FastAPI backend -> Supabase
            try:
                response = requests.post("http://localhost:8000/incident", json={
                    "camera_id": "CAM_001",
                    "location": "Camera-01 / Main Road Junction",
                    "frame_number": e["frame"],
                    "accident_type": e["type"],
                    "severity_score": severity["score"],
                    "severity_level": severity["level"],
                    "response_level": e["dispatch"]["response_level"],
                    "notify": e["dispatch"]["notify"],
                    "incident_summary": e["dispatch"]["incident_summary"],
                })
                print(f"      BACKEND: {response.json()}")
            except Exception as ex:
                print(f"      BACKEND ERROR: {ex}")

            all_events.append(e)

    cap.release()
    print(f"\nDone. Processed {frame_num} frames.")
    print(f"Total accident events detected: {len(all_events)}")
    for e in all_events:
        sev = e.get("severity", {})
        dispatch = e.get("dispatch", {})
        print(f"  - Frame {e['frame']}: {e['type']} ({e['objects'][0]} <-> {e['objects'][1]}) "
              f"| Severity: {sev.get('score', 'N/A')}/100 -> {sev.get('level', 'N/A')} "
              f"| Notify: {dispatch.get('notify', [])}")


if __name__ == "__main__":
    process_video("transitguard/mock_data/test_video.mp4")
