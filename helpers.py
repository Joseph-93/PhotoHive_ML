import json
from typing import Dict, Tuple, List

# Helper function to get the bounding box as a tuple
def get_box_coordinates(box: Dict) -> Tuple:
    return (box['Left'], box['Top'], box['Left'] + box['Width'], box['Top'] + box['Height'])


def s3_image_key_from_id(id: str) -> str:
    key = 'unwatermarked' + id + '.jpg'
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
        query = file.read().strip()
    return query
