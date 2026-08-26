from ultralytics import YOLO
import cv2


model = YOLO("best (16).pt")


video_path = "Watch Video Terrible potholes in Rangoon Road - Northern Natal News (360p, h264).mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video")
    exit()



unique_pothole_ids = set()



while True:

    ret, frame = cap.read()

    if not ret:
        break

    # YOLO + BYTE TRACK
    results = model.track(
        frame,
        conf=0.70,
        imgsz=640,
        tracker="bytetrack.yaml",
        persist=True,
        verbose=False
    )



    if results[0].boxes.id is not None:

        track_ids = (
            results[0]
            .boxes
            .id
            .int()
            .cpu()
            .tolist()
        )

        for track_id in track_ids:

            unique_pothole_ids.add(track_id)

   

    current_count = len(
        unique_pothole_ids
    )

    # Draw detections
    annotated_frame = results[0].plot()

    # Display count
    cv2.putText(
        annotated_frame,
        f"Total Potholes: {current_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Show video
    cv2.imshow(
        "Pothole Detection",
        annotated_frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break



cap.release()
cv2.destroyAllWindows()

print("--------------------------------")
print("Total Unique Potholes:", len(unique_pothole_ids))
print("--------------------------------")