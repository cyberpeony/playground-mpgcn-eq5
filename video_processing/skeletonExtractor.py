from ultralytics import YOLO #type: ignore
import cv2
import csv
import numpy as np
import os

def normalizeKeypoints(keypoints):
    """
    Normalize pose keypoints:
      - Translate hips midpoint to origin
      - Scale by torso length (neck–hip distance)
    """
    keypoints = np.array(keypoints, dtype=float)  
    if keypoints.shape[0] < 13:
        return np.full((17, 2), np.nan)

    LEFT_HIP, RIGHT_HIP = 11, 12
    LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6


    hip = np.nanmean(keypoints[[LEFT_HIP, RIGHT_HIP]], axis=0)
    neck = np.nanmean(keypoints[[LEFT_SHOULDER, RIGHT_SHOULDER]], axis=0)


    if np.any(np.isnan(hip)) or np.any(np.isnan(neck)):
        scale = 1.0
    else:
        scale = np.linalg.norm(neck - hip)
        if scale < 1e-6:
            scale = 1.0

    normalized = (keypoints - hip) / scale

    return normalized
# Load model

def skeletonExtractor(video_path: str, scene_name: str):

    fps_process = 12           # target processing fps
    T = 48                     # frames per model window
    model = YOLO("yolov8m-pose.pt")

    video_dir = os.path.join(video_path, f"{scene_name}.mp4")
    cap = cv2.VideoCapture(video_dir)
    if not cap.isOpened():
        print(f"Error opening video file: {video_dir}")
        return
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(round(fps_video / fps_process))
    cap.release()

    # Run inference
    results = model.predict(source=video_dir, stream=True, verbose=False)

    # Create CSV file

    csv_directory = "/home/sformador/equipo5/Socio-Formador-IA-Avanzada/video_processing/skeletonCSV"

    try:
        csv_filename = os.path.join(csv_directory, f"{scene_name}.csv")
        with open(csv_filename, mode="w", newline="") as f:
            writer = csv.writer(f)

            header = ["frame", "person_id"]

            for i in range(17):
                header += [f"x{i}", f"y{i}"]
            writer.writerow(header)

            frame_id = 0
            
            for r in results:
                if frame_id % frame_interval != 0:
                    frame_id += 1
                    continue
                keypoints = r.keypoints
                if not hasattr(r, "keypoints") or r.keypoints is None:
                    frame_id += 1
                    continue

                data = keypoints.xy.cpu().numpy() #type: ignore

                data_normalized = []
                for person_keypoints in data:
                    normalized = normalizeKeypoints(person_keypoints)
                    data_normalized.append(normalized)
                
                for person_id, person_keypoints in enumerate(data_normalized):
                    if person_id >= 3:
                        break

                    row = [frame_id, person_id]
                    for (x, y) in person_keypoints:
                        row += [x, y]
                    writer.writerow(row)

                frame_id += 1
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return
    
    cv2.destroyAllWindows()
    print(f"✅ Saved pose data to {csv_filename}")
    return 