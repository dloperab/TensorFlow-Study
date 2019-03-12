import sys
sys.path.append("C:/tensorflow/models/research/")
sys.path.append("C:/tensorflow/models/research/object_detection")

import numpy as np
import os
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
import zipfile
import cv2
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Define the video stream
cap = cv2.VideoCapture(0)  # Change only if you have more than one webcams

# construct paths
# base path where we will save our models
PATH_TO_OBJ_DETECTION = 'C:/tensorflow/models/research/object_detection'
# What model to download.
# Models can bee found here: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
MODEL_NAME = 'ssd_inception_v2_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = PATH_TO_OBJ_DETECTION + '/' + MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = PATH_TO_OBJ_DETECTION + '/data/mscoco_label_map.pbtxt'
# path to download model
DESTINATION_MODEL_TAR_PATH = PATH_TO_OBJ_DETECTION + '/' + MODEL_FILE
DOWNLOAD_BASE_URL = DOWNLOAD_BASE + '/' + MODEL_FILE

# print var paths info
print("[INFO] VARS PATHS:")
print(" PATH_TO_OBJ_DETECTION = {}".format(PATH_TO_OBJ_DETECTION))
print(" MODEL_FILE = {}".format(MODEL_FILE))
print(" PATH_TO_CKPT = {}".format(PATH_TO_CKPT))
print(" PATH_TO_LABELS = {}".format(PATH_TO_LABELS))
print(" DESTINATION_MODEL_TAR_PATH = {}".format(DESTINATION_MODEL_TAR_PATH))
print(" DOWNLOAD_BASE_URL = {}".format(DOWNLOAD_BASE_URL))
print("**********")

# Number of classes to detect
NUM_CLASSES = 90

# Download model if need it
if not os.path.exists(DESTINATION_MODEL_TAR_PATH):
    print("[INFO] downloading model '{}'...".format(MODEL_NAME))
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE_URL, DESTINATION_MODEL_TAR_PATH)
    tar_file = tarfile.open(DESTINATION_MODEL_TAR_PATH)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            print("[INFO] extracting model '{}'...".format(MODEL_NAME))
            tar_file.extract(file, PATH_TO_OBJ_DETECTION)

# Load a (frozen) Tensorflow model into memory.
print("[INFO] loading frozen model...")
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
print("[INFO] loading labels map...")
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

# Detection
print("[INFO] detecting from webcam...")
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:
            # Read frame from camera
            ret, image_np = cap.read()
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Extract image tensor
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Extract detection boxes
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Extract detection scores
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            # Extract detection classes
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            # Extract number of detectionsd
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                min_score_thresh=.5,
                use_normalized_coordinates=True,
                line_thickness=8)

            # Display output
            cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))

            # Return found objects
            print([category_index.get(i) for i in classes[0]])
            print(boxes.shape)
            print(num_detections)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break