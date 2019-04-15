# TensorFlow - TensorFlow Object Detection API

## Setup - TensorFlow Object Detection API

* In folder [/setup](setup) you'll find the [instructions](setup/tf_object_detection_api_setup_w10.pdf) **[Spanish]** for installing de API in Windows 10.

## PoCs - TensorFlow Object Detection API

* **PoCs - Images Object Detection:**
  * **Run Windows:**
    * Images detection - Basic: [00_basic_obj_det.py](src/images/windows/00_basic_obj_det.py).
  * **Run Linux:**
    * Images detection - Basic: [00_basic_obj_det.py](src/images/linux/00_basic_obj_det.py).

* **PoCs - Webcam Object Detection:**
  * **Run Windows:**
    * Webcam detection - Basic: [00_basic_obj_det.py](src/webcam/00_basic_obj_det.py).
  * **Credits:** [Detect Objects Using Your Webcam](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/camera.html).

* **PoCs - Video Object Detection:**
  * **Run Windows:**
    * Video detection - Basic: [00_basic_obj_det.py](src/videos/windows/00_basic_obj_det.py).
    * Video detection - Multi Processing and Threading: [01_multi_processing_threading_obj_det.py](src/videos/windows/01_multi_processing_threading_obj_det.py).
  * **Run Linux:**
    * Video detection - Basic: [00_basic_obj_det.py](src/videos/linux/00_basic_obj_det.py).
  * **Credits:**
    * [Increasing webcam FPS with Python and OpenCV](https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/).
    * [Building a Real-Time Object Recognition App with Tensorflow and OpenCV](https://towardsdatascience.com/building-a-real-time-object-recognition-app-with-tensorflow-and-opencv-b7a2b4ebdc32).
    * [Real-time and video processing object detection using Tensorflow, OpenCV and Docker](https://towardsdatascience.com/real-time-and-video-processing-object-detection-using-tensorflow-opencv-and-docker-2be1694726e5).

* **PoCs - Custom Datasets:**
  * [Raccoon Detector](src/raccoon_detector/README.md).