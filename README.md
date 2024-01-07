# Face Detection and Age-Gender Estimation

## Introduction
This Python program utilizes OpenCV to detect faces in images and estimate their gender and age (with potential inaccuracies). The face detection is based on the haarcascade_frontalface_default.xml, and age-gender estimation uses pretrained models.

## Installation

### Required Packages
1. Download [haarcascade_frontalface_default.xml](https://github.com/kipr/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml)
2. For age and gender detection, download:
   - [age_deploy.prototxt](https://github.com/smahesh29/Gender-and-Age-Detection/blob/master/age_deploy.prototxt)
   - [gender_deploy.prototxt](https://github.com/smahesh29/Gender-and-Age-Detection/blob/master/gender_deploy.prototxt)
   - [age_net.caffemodel](https://github.com/smahesh29/Gender-and-Age-Detection/blob/master/age_net.caffemodel)
   - [gender_net.caffemodel](https://github.com/smahesh29/Gender-and-Age-Detection/blob/master/gender_net.caffemodel)
   Place these files in the 'model' directory.

### Running the Code
1. Update the following line in the code with the path to your image:
   ```python
   image = cv2.imread('your_image_path.png')
2. Run the detector.py file
   ```python
   python3 detector.py
