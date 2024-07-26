import cv2
import json
import torch
import numpy as np
from tqdm import tqdm
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.structures import Boxes, Instances
import os

# Configure Detectron2
def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # Lower threshold for more sensitive detection
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg

# Process a single frame and extract bounding boxes
def process_frame(frame, predictor):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    outputs = predictor(rgb_frame)
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy()  # Get bounding boxes in format [x1, y1, x2, y2]
    keypoints = instances.pred_keypoints.to("cpu").numpy()  # Get keypoints

    return boxes, keypoints

# Save the frame with bounding boxes
def save_frame_with_boxes(frame, boxes, frame_number, folder_name):
    boxes = np.array(boxes, dtype=np.float32)
    instances = Instances(image_size=frame.shape[:2])
    instances.pred_boxes = Boxes(boxes)
    instances.scores = torch.tensor([1.0] * len(boxes))  # Dummy scores for visualization
    instances.pred_classes = torch.zeros(len(boxes), dtype=torch.int64)  # Dummy classes

    v = Visualizer(frame[:, :, ::-1], metadata=None, instance_mode=ColorMode.IMAGE)
    out = v.draw_instance_predictions(instances)
    output_frame = out.get_image()[:, :, ::-1]
    cv2.imwrite(os.path.join(folder_name, f"frame_{frame_number:04d}.png"), output_frame)

# Save the raw frame
def save_raw_frame(frame, frame_number, folder_name):
    cv2.imwrite(os.path.join(folder_name, f"frame_{frame_number:04d}.png"), frame)

# Create folders if they don't exist
def create_folders(folders):
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

def main(video_path):
    # Open video file
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames
    predictor = DefaultPredictor(setup_cfg())

    # Create folders
    raw_folder = "raw_frames"
    bbox_folder = "bbox_detectron"
    create_folders([raw_folder, bbox_folder])

    # Process frames
    bounding_boxes_data = []
    with tqdm(total=total_frames, desc="Processing Frames", unit="frame") as pbar:
        frame_number = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Save raw frame
            save_raw_frame(frame, frame_number, raw_folder)

            # Process frame
            boxes, _ = process_frame(frame, predictor)

            # Save frame with bounding boxes
            if len(boxes) > 0:
                save_frame_with_boxes(frame, boxes, frame_number, bbox_folder)

            # Save bounding boxes data
            frame_bboxes = [{'x1': float(box[0]), 'y1': float(box[1]), 'x2': float(box[2]), 'y2': float(box[3])} for box in boxes]
            if frame_bboxes:
                bounding_boxes_data.append({
                    "frame": frame_number,
                    "bboxes": frame_bboxes
                })

            frame_number += 1
            pbar.update(1)  # Update progress bar by one frame

    cap.release()
    cv2.destroyAllWindows()

    # Save bounding boxes to JSON file
    with open('detectron_bbox.json', 'w') as f:
        json.dump(bounding_boxes_data, f, indent=4)

if __name__ == "__main__":
    video_path = '10.mp4'  # Path to your input video
    main(video_path)
