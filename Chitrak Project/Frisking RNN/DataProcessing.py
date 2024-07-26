# Pass the JSON file and the images folder which you exported after annotating the data in COCO format. 
# This would give another JSON file which could be used to train the RNN model. 


import json
import cv2
import torch
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import warnings

warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.")

# Make the data look good
def firstScript(data):
    desired_categories = [
        {"id": 1, "name": "Frisked", "supercategory": ""},
        {"id": 2, "name": "Not Frisked", "supercategory": ""}
    ]
    category_mapping = {3: 1, 4: 2}
    data['categories'] = desired_categories
    for ann in data['annotations']:
        if ann['category_id'] in category_mapping:
            ann['category_id'] = category_mapping[ann['category_id']]
    return data

# Make the bbox data into integers
def secondScript(data):
    for annotation in data["annotations"]:
        annotation["bbox"] = [int(coord) for coord in annotation["bbox"]]
    return data

# Detectron Model
def extract_keypoints(image, bbox, predictor):
    x, y, w, h = bbox
    cropped_image = image[y:y+h, x:x+w]
    outputs = predictor(cropped_image)
    keypoints = outputs["instances"].pred_keypoints.cpu().numpy() if outputs["instances"].has("pred_keypoints") else []
    return keypoints.tolist()  # Convert numpy array to list

# Detectron Model
def thirdScript(data, image_base_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = DefaultPredictor(cfg)

    images = data["images"]
    annotations = data["annotations"]
    categories = {cat["id"]: cat["name"] for cat in data["categories"]}

    new_data = []

    for annotation in annotations:
        image_id = annotation["image_id"]
        image_info = next(img for img in images if img["id"] == image_id)
        image_path = f"{image_base_path}/{image_info['file_name']}"
        print(image_path, 'loading')

        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        bbox = annotation["bbox"]
        keypoints = extract_keypoints(image, bbox, predictor)
        category_id = annotation["category_id"]
        label = categories[category_id]

        new_data.append({
            "image_id": image_id,
            "image_path": image_path,
            "keypoints": keypoints,
            "label": label
        })

    return new_data

# Beautify the JSON data
def fourthScript(data):
    beautified_data = []
    for record in data:
        beautified_record = {
            "image_id": record["image_id"],
            "image_path": record["image_path"],
            "keypoints": record["keypoints"],
            "label": record["label"]
        }
        beautified_data.append(beautified_record)
    return beautified_data

# Keep the first set of keypoints only
def fifthScript(data):
    for entry in data:
        if 'keypoints' in entry and isinstance(entry['keypoints'], list) and len(entry['keypoints']) > 0:
            entry['keypoints'] = entry['keypoints'][0]  # Keep only the first set of keypoints
    return data

# Final Organization of the points to feed to RNN
def sixthScript(data):
    transformed_data = []
    for sequence in data:
        transformed_sequence = {
            "keypoints_sequence": sequence["keypoints"],
            "label": sequence["label"]
        }
        transformed_data.append(transformed_sequence)
    return transformed_data

def load_json(input_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    return data

def write_json_to_file(data, output_file):
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

def main(input_file, output_file, image_folder):
    data = load_json(input_file)
    data = firstScript(data)                    # Proper Indent 
    data = secondScript(data)                   # Make bbox Integers
    data = firstScript(data)                    # Proper Indent
    data = thirdScript(data, image_folder)      # Detectron Model
    data = fourthScript(data)                   # Beautify / Indent 
    data = fifthScript(data)                    # Remove Duplicates
    data = sixthScript(data)                    # Final Organize 

    write_json_to_file(data, output_file)

# Example usage
input_file = 'instances_default.json'
output_file = 'data.json'
image_folder = 'Images_Folder'
main(input_file, output_file, image_folder)
