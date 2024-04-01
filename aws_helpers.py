import os
import boto3
from typing import Iterator, Dict
import pymysql
from PIL import Image
from helpers import s3_image_key_from_id


db_name = os.environ.get('DB_NAME')
db_user = os.environ.get('DB_USER')
db_password = os.environ.get('DB_PASSWORD')
db_host = os.environ.get('DB_HOST')
db_port = int(os.environ.get('DB_PORT'))
s3_bucket_name = os.environ.get('S3_BUCKET_NAME')

s3 = boto3.client('s3')
rekognition = boto3.client('rekognition')


def get_photo_from_s3(id: str) -> Image:
    key = s3_image_key_from_id(id)
    response = s3.get_object(Bucket=s3_bucket_name, Key=key)
    image = Image.open(response['Body'])
    return image
    

def get_rekognition_data(id: str, max_labels: int, max_faces: int) -> Dict:
    # Setup
    key = s3_image_key_from_id(id)
    object_for_rekognition = {'S3Object': {'Bucket': s3_bucket_name, 'Name': key}}

    # get rekognition data into a dictionary that is non-nested
    label_response = rekognition.detect_labels(Image=object_for_rekognition, MaxLabels=max_labels)
    people_and_bikes = [label for label in label_response['Labels'] if label['Name'] in ['Person', 'Bicycle']]

    # Detect faces to get detailed facial analysis, only keep the faces with highest ML confidence level
    face_response = rekognition.detect_faces(Image=object_for_rekognition, Attributes=['ALL'])
    faces = face_response['FaceDetails']
    face_details = sorted(faces, key=lambda x: x['Confidence'], reverse=True)[:max_faces]

    # Prepare the data to return
    rekognition_data = {
        'PeopleAndBikes': people_and_bikes,
        'FaceDetails': face_details
    }

    return rekognition_data


def run_large_query(query: str) -> Iterator[Dict]:
    # Initialize connection and cursor
    chunk_size = 1000
    conn = pymysql.connect(user=db_user,
                    password=db_password,
                    host=db_host,
                    database=db_name,
                    unix_socket=None,
                    port=db_port)
    cursor = conn.cursor()

    try:
        cursor = conn.cursor()
        cursor.execute(query)

        # Get column names
        columns = [col[0] for col in cursor.description]

        # Yield rows as dictionaries
        while True:
            rows = cursor.fetchmany(chunk_size)
            if not rows:
                break
            for row in rows:
                yield dict(zip(columns, row))
    finally:
        # Ensure resources are released
        cursor.close()
        conn.close()

