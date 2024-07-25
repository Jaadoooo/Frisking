import os
import json
from ultralytics import YOLO
from PIL import Image, ImageDraw
from tqdm import tqdm  # Import tqdm for progress bar

# Load the custom YOLO model
model = YOLO("best.pt")

# Paths
input_folder = "raw_frames"
output_folder = "bbox_yolo"
output_json_file = os.path.join(output_folder, "../yolo_bbox.json")

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Initialize the list to store the results
results_list = []

# List all image files in the input folder
image_files = [f for f in os.listdir(input_folder) if f.endswith(".png") or f.endswith(".jpg")]

# Iterate through all images in the folder with a progress bar
for filename in tqdm(image_files, desc="Processing Frames", unit="frame"):
    image_path = os.path.join(input_folder, filename)
    
    # Predict with the model
    results = model(image_path)
    
    # Load the image
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    
    # Initialize the bounding boxes list for the current image
    bboxes = []
    
    for result in results:
        if result.boxes is not None:  # Check if boxes are detected
            box_data = result.boxes.data.cpu().numpy()  # Get bounding box data
            
            for det in box_data:
                if len(det) == 6:  # Ensure there are 6 elements
                    x1, y1, x2, y2, conf, cls = det
                    bboxes.append({
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2)
                    })
                    # Draw the bounding box on the image
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    
    # Save the image with bounding boxes in the output folder
    output_image_path = os.path.join(output_folder, filename)
    image.save(output_image_path)
    
    # Store the results in the desired format
    frame_number = int(filename.split('.')[0].replace("frame_", ""))
    results_list.append({
        "frame": frame_number,
        "bboxes": bboxes
    })

# Write results to a JSON file
with open(output_json_file, 'w') as f:
    json.dump(results_list, f, indent=4)

print(f"Bounding box data has been saved to {output_json_file}.")
