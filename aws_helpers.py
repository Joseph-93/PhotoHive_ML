import os
import json
import boto3
import logging
from botocore.exceptions import ClientError
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
rekognition = boto3.client('rekognition', region_name='us-east-2')


class FaceData:
    def __init__(self, face_details, i):
        # Define all landmark types
        all_landmark_types = ['eyeLeft', 'eyeRight', 'nose', 'mouthLeft', 'mouthRight', 
                      'leftEyeBrowLeft', 'leftEyeBrowRight', 'leftEyeBrowUp', 'rightEyeBrowLeft', 
                      'rightEyeBrowRight', 'rightEyeBrowUp', 'leftEyeLeft', 'leftEyeRight', 
                      'leftEyeUp', 'leftEyeDown', 'rightEyeLeft', 'rightEyeRight', 'rightEyeUp', 
                      'rightEyeDown', 'noseLeft', 'noseRight', 'mouthUp', 'mouthDown', 'leftPupil', 
                      'rightPupil', 'upperJawlineLeft', 'midJawlineLeft', 'chinBottom', 
                      'midJawlineRight', 'upperJawlineRight']

        # Define a flattened structure with default values
        self.data = {
            F"{i}_AgeRange_High": None,
            F"{i}_AgeRange_Low": None,
            F"{i}_Beard_Confidence": 0,
            F"{i}_Beard_Value": None,
            F"{i}_BoundingBox_Height": 0,
            F"{i}_BoundingBox_Left": 0,
            F"{i}_BoundingBox_Top": 0,
            F"{i}_BoundingBox_Width": 0,
            F"{i}_Confidence": 0,
            F"{i}_EyeDirection_Confidence": 0,
            F"{i}_EyeDirection_Pitch": 0,
            F"{i}_EyeDirection_Yaw": 0,
            F"{i}_Eyeglasses_Confidence": 0,
            F"{i}_Eyeglasses_Value": None,
            F"{i}_EyesOpen_Confidence": 0,
            F"{i}_EyesOpen_Value": None,
            F"{i}_FaceOccluded_Confidence": 0,
            F"{i}_FaceOccluded_Value": None,
            F"{i}_Gender_Confidence": 0,
            F"{i}_Gender_Value": None,
            F"{i}_MouthOpen_Confidence": 0,
            F"{i}_MouthOpen_Value": None,
            F"{i}_Mustache_Confidence": 0,
            F"{i}_Mustache_Value": None,
            F"{i}_Pose_Pitch": 0,
            F"{i}_Pose_Roll": 0,
            F"{i}_Pose_Yaw": 0,
            F"{i}_Quality_Brightness": 0,
            F"{i}_Quality_Sharpness": 0,
            F"{i}_Smile_Confidence": 0,
            F"{i}_Smile_Value": None,
            F"{i}_Sunglasses_Confidence": 0,
            F"{i}_Sunglasses_Value": None
        }

        # Initialize Emotions and Landmarks with default values
        for emotion in ['CALM', 'SURPRISED', 'CONFUSED', 'SAD', 'HAPPY', 'ANGRY', 'DISGUSTED', 'FEAR', 'UNKNOWN']:
            self.data[F"{i}_Emotions_{emotion}"] = 0
        for landmark in all_landmark_types:
            self.data[F"{i}_Landmarks_{landmark}_X"] = 0
            self.data[F"{i}_Landmarks_{landmark}_Y"] = 0

        # Update the flattened dictionary with actual values from face_details
        if face_details is not None:
            self.flatten_dict(face_details, f"{i}_")

    def flatten_dict(self, d, prefix=''):
        for key, value in d.items():
            if isinstance(value, dict):
                self.flatten_dict(value, prefix + key + '_')
            elif isinstance(value, list):
                if key == 'Emotions':
                    for item in value:
                        self.data[prefix + key + '_' + item['Type']] = item['Confidence']
                elif key == 'Landmarks':
                    for item in value:
                        self.data[prefix + key + '_' + item['Type'] + '_X'] = item['X']
                        self.data[prefix + key + '_' + item['Type'] + '_Y'] = item['Y']
            else:
                self.data[prefix + key] = value


def get_photo_from_s3(id: str) -> Image:
    key = s3_image_key_from_id(id)
    response = s3.get_object(Bucket=s3_bucket_name, Key=key)
    image = Image.open(response['Body'])
    return image
    

def get_rekognition_data(id: str, max_labels: int=10, max_faces: int=10, confidence_tolerance: float=0.95) -> Dict:
    # Setup
    key = s3_image_key_from_id(id)
    object_for_rekognition = {'S3Object': {'Bucket': s3_bucket_name, 'Name': key}}

    # get rekognition data into a dictionary that is non-nested
    try:
        label_response = rekognition.detect_labels(Image=object_for_rekognition)
    except ClientError as e:
        print(e)
    people_and_bikes = [label for label in label_response['Labels'] if label['Name'] in ['Person', 'Bicycle']]

    # Detect faces to get detailed facial analysis, only keep the faces with highest ML confidence level
    face_response = rekognition.detect_faces(Image=object_for_rekognition, Attributes=['ALL'])
    faces = face_response['FaceDetails']
    face_details = sorted(faces, key=lambda x: x['Confidence'], reverse=True)[:max_faces]

    flattened_faces = {}
    for i in range(max_faces):
        if i < len(face_details):
            flattened_faces.update(FaceData(face_details[i], i).data)
        else:
            flattened_faces.update(FaceData(None, i).data)

    # Prepare the data to return
    rekognition_data = {
        'PeopleAndBikes': people_and_bikes,
        'FaceDetails': flattened_faces,
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

