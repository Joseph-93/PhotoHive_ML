import json
from typing import Dict, Tuple, List

# Helper function to get the bounding box as a tuple
def get_box_coordinates(box: Dict) -> Tuple:
    return (box['Left'], box['Top'], box['Left'] + box['Width'], box['Top'] + box['Height'])


def s3_image_key_from_id(id: str) -> str:
    key = 'unwatermarked/' + id + '.jpg'
    return key


def pack_jsons_into_row(jsons: List[str]) -> str:
    combined_json = {}
    for json_str in jsons:
        combined_json.update(json.loads(json_str))
    combined_json_string = json.dumps(combined_json)
    return combined_json_string


# Function to read a query from a file
def read_query_from_file(file_path: str) -> str:
    with open(file_path, 'r') as file:
        # Read the query and remove any leading or trailing whitespace
        query = file.read()
    return query


def bounding_boxes_to_integers(bounding_boxes: List[Dict], height: int, width: int):
    converted_bounds = []
    for box in bounding_boxes:
        box_dict = {}
        box_dict['top'] = int(box[1]*height)
        box_dict['bottom'] = int(box[3]*height)
        box_dict['left'] = int(box[0]*width)
        box_dict['right'] = int(box[2]*width)
        converted_bounds.append(box_dict)
    return converted_bounds


def verify_data_consistency(all_rows_data):
    # Convert JSON strings to dictionaries if they are not already
    data_dicts = [json.loads(row) if isinstance(row, str) else row for row in all_rows_data]

    if not data_dicts:
        return False, "No data provided."

    # Get the keys from the first row to set the standard
    standard_keys = set(data_dicts[0].keys())
    missing_keys = []

    # Check each row against the standard
    for index, row in enumerate(data_dicts):
        row_keys = set(row.keys())
        if row_keys != standard_keys:
            # Find missing and extra keys
            missing_in_row = standard_keys - row_keys
            missing_in_standard = row_keys - standard_keys
            missing_keys.append((index, {"missing_in_row": list(missing_in_row), "missing_in_standard": list(missing_in_standard)}))

    if missing_keys:
        return False, f"Inconsistency found. Details: {missing_keys}"

    return True, "All rows are consistent."