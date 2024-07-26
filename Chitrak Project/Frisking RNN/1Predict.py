import json
import cv2
import torch
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os
import warnings

warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.")

# Initialize Detectron model for keypoint detection
def initialize_detectron_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = DefaultPredictor(cfg)
    return predictor

# Function to extract keypoints from an image using Detectron
def extract_keypoints(image, predictor):
    outputs = predictor(image)
    instances = outputs["instances"]
    if instances.has("pred_keypoints"):
        keypoints = instances.pred_keypoints.cpu().numpy()
        scores = instances.scores.cpu().numpy()
        # Select the person with the highest score
        if len(scores) > 0:
            best_index = np.argmax(scores)
            keypoints = keypoints[best_index]
            return keypoints
    return []

# Function to visualize keypoints on an image
def visualize_keypoints(image, keypoints):
    for x, y, confidence in keypoints:
        cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), -1)
    return image

# Function to process video and save keypoints and frames with keypoints
def process_video(video_path, predictor, output_dir):
    # Create directories for saving frames and keypoints
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Initialize an empty list to store keypoints sequences
    keypoints_sequences = []
    frame_index = 0
    
    # Process each frame in the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extract keypoints from the frame
        keypoints = extract_keypoints(image, predictor)
        
        if len(keypoints) > 0:
            keypoints_sequences.append(keypoints.tolist())
            
            # Visualize keypoints on the frame
            frame_with_keypoints = visualize_keypoints(frame, keypoints)
            
            # Save the frame with keypoints
            save_path = os.path.join(output_dir, f"frame_{frame_index:04d}.jpg")
            cv2.imwrite(save_path, frame_with_keypoints)
        else:
            keypoints_sequences.append([])
        
        frame_index += 1
    
    cap.release()
    
    # Save keypoints to a JSON file
    keypoints_file = os.path.join(output_dir, "keypoints.json")
    with open(keypoints_file, 'w') as f:
        json.dump(keypoints_sequences, f, indent=4)
    
    return "Processing complete"

# Example usage
if __name__ == "__main__":
    # Initialize Detectron model
    predictor = initialize_detectron_model()

    # Path to your input video
    video_path = '5.mp4'  # Replace with the actual path to your video

    # Directory to save the output frames and keypoints
    output_dir = 'output_frames'  # Replace with the desired output directory

    # Process the video and save keypoints and frames
    result = process_video(video_path, predictor, output_dir)
    print(result)
