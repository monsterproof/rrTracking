from ultralytics import YOLO
import cv2
import numpy as np
from pixel_tracker import PixelTracker 

model = YOLO("yolov8n-pose.pt")
cap = cv2.VideoCapture(0)

# Get FPS from video capture
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps is None:
    fps = 30  # Default to 30 if unable to detect
print(f"Video FPS: {fps}")

# Initialize tracker as None - will be created when first valid ROI is found
tracker = None
prev_frame = None
roi_box = None

print("Starting thorax tracking...")
print("Press 'q' to quit")
print("Press 'p' to show/save plot")
print("Press 'r' to reset tracking")

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
        conf = float(best_box.conf[0])
        
        # Keypoints
        left_shoulder = best_kpts[5]
        right_shoulder = best_kpts[6]
        
        if np.any(left_shoulder == 0) or np.any(right_shoulder == 0):
            cv2.imshow("Thorax Detection - Single Person + Confidence", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
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
        x1 = int(cx - thorax_width / 2 + thorax_width * 0.3)
        x2 = int(cx + thorax_width / 2 - thorax_width * 0.3)
        y1 = int(cy)
        y2 = int(cy + thorax_height)
        
        # Ensure coordinates are within frame bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        new_roi_box = (x1, y1, x2, y2)
        
        # Initialize tracker on first valid frame OR if ROI changed significantly
        if tracker is None or prev_frame is None:
            tracker = PixelTracker(new_roi_box, n_points=10, history_length=1800, use_bg_subtraction=False, track_shoulders=False, fps=fps)
            if tracker.find_distinctive_points(frame):
                roi_box = new_roi_box
                # Initialize shoulder tracking
                tracker.update_shoulder_position(left_shoulder, right_shoulder)
                print(f"Tracker initialized with {len(tracker.tracked_points)} points")
            else:
                print("Warning: Could not find distinctive points in ROI")
                tracker = None
        else:
            # Update shoulder position for this frame
            tracker.update_shoulder_position(left_shoulder, right_shoulder)
            
            # Update the ROI to follow the moving thorax
            tracker.update_roi(new_roi_box)
            roi_box = new_roi_box
            
            # Track points from previous frame to current frame
            if tracker.track_points(prev_frame, frame):
                pass  # Tracking successful
            else:
                # Tracking failed - reinitialize
                print("Tracking lost, reinitializing...")
                tracker = PixelTracker(new_roi_box, n_points=10, history_length=1800, use_bg_subtraction=False, track_shoulders=False, fps=fps)
                if tracker.find_distinctive_points(frame):
                    roi_box = new_roi_box
                    tracker.update_shoulder_position(left_shoulder, right_shoulder)
                else:
                    tracker = None
        
        # Draw tracked points if tracker exists
        if tracker is not None:
            frame = tracker.draw_tracked_points(frame)
        else:
            # Only draw ROI box if tracker doesn't exist (since draw_tracked_points draws it)
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
        
        # Update previous frame
        prev_frame = frame.copy()
    
    # Display frame
    cv2.imshow("Thorax Detection - Single Person + Confidence", frame)
    
    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        # Show plot
        if tracker is not None:
            tracker.plot_vertical_movement('vertical_movement.png')
        else:
            print("No tracker initialized yet")
    elif key == ord('r'):
        # Reset tracking
        tracker = None
        prev_frame = None
        print("Tracking reset")

# Final plot
if tracker is not None:
    print(f"\nFinal tracking results: {tracker.frame_count} frames")
    tracker.plot_vertical_movement('vertical_movement_final.png')
else:
    print("No tracking data collected")

cap.release()
cv2.destroyAllWindows()