import json

def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Calculate the area of both boxes
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    # Calculate the intersection area
    x1_int = max(x1_min, x2_min)
    y1_int = max(y1_min, y2_min)
    x2_int = min(x1_max, x2_max)
    y2_int = min(y1_max, y2_max)

    if x1_int < x2_int and y1_int < y2_int:
        intersection_area = (x2_int - x1_int) * (y2_int - y1_int)
    else:
        return 0.0

    # Calculate IoU
    iou = intersection_area / (area1 + area2 - intersection_area)
    return iou

def filter_bboxes(detectron_bboxes, yolo_bboxes, threshold=0.6):
    filtered_bboxes = []

    for frame in detectron_bboxes:
        frame_number = frame["frame"]
        detectron_boxes = frame["bboxes"]
        
        # Get YOLO boxes for the same frame
        yolo_boxes = next((item['bboxes'] for item in yolo_bboxes if item['frame'] == frame_number), [])

        if yolo_boxes:
            # Filter detectron boxes based on IoU with YOLO boxes
            valid_detectron_boxes = []
            for det_box in detectron_boxes:
                det_box_coords = [det_box["x1"], det_box["y1"], det_box["x2"], det_box["y2"]]
                box_valid = False
                for yolo_box in yolo_boxes:
                    yolo_box_coords = [yolo_box["x1"], yolo_box["y1"], yolo_box["x2"], yolo_box["y2"]]
                    iou = calculate_iou(det_box_coords, yolo_box_coords)
                    if iou >= threshold:
                        box_valid = True
                        break
                if box_valid:
                    valid_detectron_boxes.append(det_box)
            
            # If valid boxes found, use them
            filtered_bboxes.append({
                "frame": frame_number,
                "bboxes": valid_detectron_boxes
            })
        else:
            # No YOLO boxes detected, keep only the first Detectron box
            if detectron_boxes:
                first_box = detectron_boxes[0]
                filtered_bboxes.append({
                    "frame": frame_number,
                    "bboxes": [first_box]
                })
            else:
                filtered_bboxes.append({
                    "frame": frame_number,
                    "bboxes": []
                })

    return filtered_bboxes

def main(detectron_json, yolo_json, output_json):
    # Load bounding boxes from JSON files
    with open(detectron_json, 'r') as f:
        detectron_bboxes = json.load(f)

    with open(yolo_json, 'r') as f:
        yolo_bboxes = json.load(f)

    # Filter the bounding boxes
    filtered_bboxes = filter_bboxes(detectron_bboxes, yolo_bboxes)

    # Save filtered bounding boxes to a JSON file
    with open(output_json, 'w') as f:
        json.dump(filtered_bboxes, f, indent=4)

if __name__ == "__main__":
    detectron_json = 'detectron_bbox.json'  # Path to Detectron bounding boxes JSON
    yolo_json = 'yolo_bbox.json'  # Path to YOLO bounding boxes JSON
    output_json = 'filtered_bbox.json'  # Path to save filtered bounding boxes
    main(detectron_json, yolo_json, output_json)
