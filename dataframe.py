import json
import pandas as pd
from typing import Dict

from PhotoHive_DSP.core import get_report, set_bounding_boxes
from aws_helpers import get_photo_from_s3, get_rekognition_data, run_large_query
from helpers import get_box_coordinates, pack_jsons_into_row, read_query_from_file


def flatten_rekognition_data(data: Dict) -> Dict:
    flattened_data = {}
    max_faces = 10  # Maximum number of faces to process

    # Flatten FaceDetails
    for i, face in enumerate(data.get('FaceDetails', [])[:max_faces]):
        prefix = f'Face_{i+1}'
        for key, value in face.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flattened_data[f'{prefix}_{key}_{sub_key}'] = sub_value
            elif isinstance(value, list):
                for j, item in enumerate(value):
                    for sub_key, sub_value in item.items():
                        flattened_data[f'{prefix}_{key}_{j+1}_{sub_key}'] = sub_value
            else:
                flattened_data[f'{prefix}_{key}'] = value

    # Flatten Labels (assuming similar processing is needed)
    for i, label in enumerate(data.get('Labels', [])):
        prefix = f'Label_{i+1}'
        for key, value in label.items():
            if key in ['Instances', 'Parents', 'Aliases']:  # Handle nested structures
                for j, item in enumerate(value):
                    for sub_key, sub_value in item.items():
                        flattened_data[f'{prefix}_{key}_{j+1}_{sub_key}'] = sub_value
            else:
                flattened_data[f'{prefix}_{key}'] = value

    # Remove unwanted keys related to background and foreground
    unwanted_keys = ['Background_DominantColors', 'Foreground_DominantColors']
    for key in unwanted_keys:
        flattened_data.pop(key, None)

    return flattened_data


def get_bounding_boxes_from_rekognition_data(rekognition_data: Dict) -> Dict:
    # Helper function to merge two bounding boxes
    def merge_boxes(box1, box2):
        left = min(box1[0], box2[0])
        top = min(box1[1], box2[1])
        right = max(box1[2], box2[2])
        bottom = max(box1[3], box2[3])
        return left, top, right, bottom

    # Check if two boxes overlap significantly (more than 50% overlap for this example)
    def is_significant_overlap(box1, box2):
        left1, top1, right1, bottom1 = box1
        left2, top2, right2, bottom2 = box2

        # Calculate the intersecting rectangle
        intersect_left = max(left1, left2)
        intersect_top = max(top1, top2)
        intersect_right = min(right1, right2)
        intersect_bottom = min(bottom1, bottom2)

        # If there is no intersection, return False
        if intersect_right <= intersect_left or intersect_bottom <= intersect_top:
            return False

        # Calculate intersection area
        intersect_area = (intersect_right - intersect_left) * (intersect_bottom - intersect_top)
        box1_area = (right1 - left1) * (bottom1 - top1)
        box2_area = (right2 - left2) * (bottom2 - top2)

        # Check if the intersection is significant in either of the bounding boxes
        return intersect_area > 0.5 * box1_area or intersect_area > 0.5 * box2_area

    # Extract bounding boxes for people and bikes
    people_boxes = []
    bike_boxes = []

    for label in rekognition_data.get('Labels', []):
        if label['Name'] == 'Person':
            for instance in label.get('Instances', []):
                people_boxes.append(get_box_coordinates(instance['BoundingBox']))
        elif label['Name'] == 'Bicycle':
            for instance in label.get('Instances', []):
                bike_boxes.append(get_box_coordinates(instance['BoundingBox']))

    # Merge overlapping boxes
    for bike_box in bike_boxes:
        for person_box in people_boxes:
            if is_significant_overlap(bike_box, person_box):
                merged_box = merge_boxes(bike_box, person_box)
                people_boxes.append(merged_box)
                people_boxes.remove(person_box)
                bike_boxes.remove(bike_box)
                break

    # Combine all boxes and convert to the required format
    all_boxes = people_boxes + bike_boxes
    bounding_boxes = {
        "top": min(box[1] for box in all_boxes),
        "bottom": max(box[3] for box in all_boxes),
        "left": min(box[0] for box in all_boxes),
        "right": max(box[2] for box in all_boxes)
    }

    return bounding_boxes


def create_data_frame(query_file_path: str, max_labels: int = 10, max_faces: int = 10) -> pd.DataFrame:
    if query_file_path is None or query_file_path == "" or not query_file_path.__contains__(".txt"):
        print("ERROR: query_file_path is required and must be a .txt file.")
        return None
    all_rows_data = []
    # Get query from query.txt
    query = read_query_from_file(query_file_path)

    # For every row in the query
    for db_data in run_large_query(query):
        # Get Recoknition data from AWS
        id = db_data["photoid"]
        unflattened_rekognition_data = get_rekognition_data(id, max_labels, max_faces)
        rekognition_data = flatten_rekognition_data(unflattened_rekognition_data)

        # Get bounding boxes object set (from Rekognition data)
        bounding_boxes = get_bounding_boxes_from_rekognition_data(rekognition_data)
        crop_bounds = set_bounding_boxes(bounding_boxes)

        # Get the PhotoHive_DSP report
        image = get_photo_from_s3(id)
        report = get_report(image, salient_characters=crop_bounds, coverage_thresh=0.97, downsample_rate=3, fft_streak_thresh=1.15)

        # Generate visual reports (DEBUG ONLY)
        report.image = image
        report.generate_color_palette_image()
        report.bounding_boxes = bounding_boxes
        report.display_all()

        # Pack JSONs together
        report_json = report.to_json()
        rekognition_json = json.dumps(rekognition_data)
        total_json_row = pack_jsons_into_row([db_data, report_json, rekognition_json])
        all_rows_data.append(total_json_row)

    # Create and return dataframe
    data_frame = pd.read_json(all_rows_data)
    return data_frame