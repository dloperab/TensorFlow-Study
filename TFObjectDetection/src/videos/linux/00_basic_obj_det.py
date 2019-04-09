import sys
import os
sys.path.append(os.environ['TF_RESEARCH'])

import numpy as np
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
import cv2

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# What model to download
# Models can bee found here: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28'

# construct paths
# base path where we will save our models
OBJECT_DETECTION_FOLDER = 'tensorflow/models/research/object_detection'
PATH_TO_OBJ_DETECTION = os.path.join(os.path.expanduser('~'), OBJECT_DETECTION_FOLDER)
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = PATH_TO_OBJ_DETECTION + '/' + MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = PATH_TO_OBJ_DETECTION + '/data/mscoco_label_map.pbtxt'
# path to download model
DESTINATION_MODEL_TAR_PATH = PATH_TO_OBJ_DETECTION + '/' + MODEL_FILE
DOWNLOAD_BASE_URL = DOWNLOAD_BASE + '/' + MODEL_FILE

# Number of classes to detect
NUM_CLASSES = 90

# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def download_model():
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

# Detection
def detect_objects(image_np, sess, detection_graph):    
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Extract image tensor
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    
    # Extract detection boxes
    # Each box represents a part of the image where a particular object was detected
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    
    # Extract detection scores
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')                
    # Extract detection classes
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    # Extract number of detections
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    
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

    return image_np     

def worker(cap):
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    while True:
        ret, image_np = cap.read()
        detected_image = detect_objects(image_np, sess, detection_graph)
        
        cv2.imshow('video detection', cv2.resize(detected_image, (800, 600)))

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    sess.close()

if __name__ == '__main__':
    # Define the video stream
    cap = cv2.VideoCapture("../../data/videos/traffic.mp4")

    # download_model()
    worker(cap)
