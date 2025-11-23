from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolov8n-pose.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    if len(results.boxes) > 0:
        # 1) Pick the highest-confidence person
        confidences = results.boxes.conf.cpu().numpy()
        best_idx = np.argmax(confidences)

        best_box = results.boxes[best_idx]
        best_kpts = results.keypoints[best_idx].xy[0].cpu().numpy()
        conf = float(best_box.conf[0])  # <-- confidence value

        # Keypoints
        left_shoulder = best_kpts[5]
        right_shoulder = best_kpts[6]

        if np.any(left_shoulder == 0) or np.any(right_shoulder == 0):
            cv2.imshow("Thorax", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        shoulder_mid = (left_shoulder + right_shoulder) / 2

        # Hip logic
        if np.any(best_kpts[11] == 0) or np.any(best_kpts[12] == 0):
            shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
            hip_mid = shoulder_mid + np.array([0, shoulder_width * 1.2])
        else:
            hip_mid = (best_kpts[11] + best_kpts[12]) / 2

        # Thorax geometry
        shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
        thorax_width = shoulder_width * 1.2
        thorax_height = (hip_mid[1] - shoulder_mid[1]) * 0.5

        cx, cy = shoulder_mid
        x1 = int(cx - thorax_width / 2)
        x2 = int(cx + thorax_width / 2)
        y1 = int(cy)
        y2 = int(cy + thorax_height)

        # Draw thorax box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw confidence above box
        cv2.putText(
            frame,
            f"{conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        # Shoulder markers
        cv2.circle(frame, (int(left_shoulder[0]), int(left_shoulder[1])), 4, (0, 0, 255), -1)
        cv2.circle(frame, (int(right_shoulder[0]), int(right_shoulder[1])), 4, (0, 0, 255), -1)

    cv2.imshow("Thorax Detection - Single Person + Confidence", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
