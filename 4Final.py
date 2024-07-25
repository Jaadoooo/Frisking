import cv2
import json
import os
import numpy as np
import math
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from tqdm import tqdm
import torch

# Configure Detectron2
def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg

# Calculate angle between three points
def calculate_angle(p1, p2, p3):
    angle = math.degrees(math.atan2(p3[1] - p2[1], p3[0] - p2[0]) - math.atan2(p1[1] - p2[1], p1[0] - p2[0]))
    angle = abs(angle)
    if angle > 180:
        angle = 360 - angle
    return angle

# Check if a keypoint lies within a bounding box
def keypoint_in_bbox(keypoint, bbox):
    x, y, _ = keypoint
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]

# Process a single frame and extract keypoints
def process_frame(frame, predictor, frame_number, bbox_data, keypoints_folder):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    outputs = predictor(rgb_frame)
    instances = outputs["instances"].to("cpu")
    keypoints = instances.pred_keypoints.to("cpu").numpy()
    
    # Extract bounding boxes for the current frame
    frame_bboxes = next((item['bboxes'] for item in bbox_data if item['frame'] == frame_number), [])
    
    # Filter keypoints within bounding boxes
    filtered_keypoints = []
    for bbox in frame_bboxes:
        bbox_coords = [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]
        for kp in keypoints:
            if all(keypoint_in_bbox(point, bbox_coords) for point in kp):
                filtered_keypoints.append(kp)
                
    # Draw keypoints and check for frisking
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_frisked = False
    
    for kp in filtered_keypoints:
        for i, keypoint in enumerate(kp):
            x, y, _ = keypoint
            cv2.circle(frame_rgb, (int(x), int(y)), 3, (0, 0, 255), -1)
        
        # Draw lines and angles
        if len(kp) >= 9:
            left_shoulder = kp[5]
            right_shoulder = kp[6]
            left_elbow = kp[7]
            right_elbow = kp[8]

            angle1 = calculate_angle(right_elbow, right_shoulder, left_shoulder)
            angle2 = calculate_angle(left_elbow, left_shoulder, right_shoulder)

            # Draw lines
            cv2.line(frame_rgb, (int(left_elbow[0]), int(left_elbow[1])), (int(left_shoulder[0]), int(left_shoulder[1])), (255, 0, 0), 2)
            cv2.line(frame_rgb, (int(left_shoulder[0]), int(left_shoulder[1])), (int(right_shoulder[0]), int(right_shoulder[1])), (255, 0, 0), 2)
            cv2.line(frame_rgb, (int(right_shoulder[0]), int(right_shoulder[1])), (int(right_elbow[0]), int(right_elbow[1])), (255, 0, 0), 2)

            if 140 <= angle1 <= 190 and 140 <= angle2 <= 190:
                frame_frisked = True
    
    # Add text for frisking status
    text = "Frisked" if frame_frisked else "Not Frisked"
    color = (0, 255, 0) if frame_frisked else (0, 0, 255)
    cv2.putText(frame_rgb, text, (10, frame_rgb.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Save the frame with keypoints
    cv2.imwrite(os.path.join(keypoints_folder, f"frame_{frame_number:04d}.png"), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    
    return filtered_keypoints, frame_frisked

def main(video_path, filtered_bboxes_json, output_keypoints_json, output_video_path):
    # Open video file
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Load filtered bounding boxes
    with open(filtered_bboxes_json, 'r') as f:
        filtered_bboxes_data = json.load(f)
    
    # Setup Detectron2
    predictor = DefaultPredictor(setup_cfg())
    
    # Create folders
    keypoints_folder = "Filtered_keypoints_images"
    if not os.path.exists(keypoints_folder):
        os.makedirs(keypoints_folder)
    
    # Process frames
    keypoints_data = []
    with tqdm(total=total_frames, desc="Processing Frames", unit="frame") as pbar:
        frame_number = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            filtered_keypoints, frame_frisked = process_frame(frame, predictor, frame_number, filtered_bboxes_data, keypoints_folder)
            keypoints_data.append({
                "frame": frame_number,
                "keypoints": [kp.tolist() for kp in filtered_keypoints],  # Convert numpy arrays to lists
                "frisked": frame_frisked
            })
            
            frame_number += 1
            pbar.update(1)
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Save keypoints to JSON file
    with open(output_keypoints_json, 'w') as f:
        json.dump(keypoints_data, f, indent=4)
    
    # Create output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    for i in range(total_frames):
        img_path = os.path.join(keypoints_folder, f"frame_{i:04d}.png")
        frame = cv2.imread(img_path)
        out.write(frame)
    
    out.release()

if __name__ == "__main__":
    video_path = '7.mp4'  # Path to your input video
    filtered_bboxes_json = 'filtered_bbox.json'  # Path to filtered bounding boxes JSON
    output_keypoints_json = 'keypoints.json'  # Path to save extracted keypoints
    output_video_path = 'output_video.mp4'  # Path to save output video
    main(video_path, filtered_bboxes_json, output_keypoints_json, output_video_path)
