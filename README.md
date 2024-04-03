# PhotoHive-ML

PhotoHive-ML is a machine learning preprocessing tool designed to streamline the analysis of image data, integrating various AWS services and the PhotoHive_DSP library. This tool facilitates the extraction of meaningful insights from large image datasets, preparing the data for machine learning models.

## Features

- Integration with AWS Rekognition for image analysis and feature extraction.
- Utilizes the PhotoHive_DSP library for advanced image processing tasks.
- Efficient handling of large image datasets from an SQL database.
- Generation of comprehensive data frames ready for machine learning applications.

## Requirements

- AWS account with access to Rekognition and S3 services.
- MySQL or compatible SQL database for image metadata.
- Python 3 environment with necessary libraries installed (Pandas, Boto3, PyMySQL, Pillow).

## Setup and Installation

1. Clone the repository to your local machine.

```bash
git clone https://github.com/Joseph-93/PhotoHive_ML.git
```

2. Navigate to the cloned directory.

```bash
cd PhotoHive-ML
```

3. Ensure that you have the PhotoHive_DSP library installed and configured as per the instructions available at [PhotoHive_DSP repository](https://github.com/Joseph-93/PhotoHive_DSP).

4. Set up your Python environment and install the required dependencies.

```bash
pip install -r requirements.txt
```


5. Configure your AWS credentials and ensure access to the necessary AWS services (Rekognition, S3).

6. Set up your SQL database and ensure it is accessible from your environment.

## Usage

The tool is designed to be run as a Python script, where it processes images from the specified SQL database, extracts features using AWS Rekognition, and utilizes the PhotoHive_DSP library for additional image processing.

To start processing your image data and generating a data frame for machine learning:

```bash
python main.py
```

Ensure that `main.py` is configured to connect to your SQL database and AWS services correctly.

## Contributing

Contributions to the PhotoHive-ML project are welcome. Please ensure to follow the standard pull request process for submitting your contributions.

---

For detailed information on the setup and usage of the PhotoHive_DSP library, refer to the [official PhotoHive_DSP GitHub repository](https://github.com/Joseph-93/PhotoHive_DSP).