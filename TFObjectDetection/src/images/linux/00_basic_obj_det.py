import sys
import os
sys.path.append(os.environ['TF_RESEARCH'])

import numpy as np
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
import cv2
from PIL import Image

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# construct paths
# base path where we will save our models
OBJECT_DETECTION_FOLDER = 'tensorflow/models/research/object_detection'
PATH_TO_OBJ_DETECTION = os.path.join(os.path.expanduser('~'), OBJECT_DETECTION_FOLDER)
# What model to download.
# Models can bee found here: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28'
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
# Images
PATH_TO_TEST_IMAGES_DIR = '../../data/images/'
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3)]
# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  
  return output_dict

print("[INFO] detecting from images...")
counter = 1
for image_path in TEST_IMAGE_PATHS:
  print("[INFO] detecting image: {}".format(image_path))
  
  image = Image.open(image_path)
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = load_image_into_numpy_array(image)
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_np, axis=0)
  # Actual detection.
  output_dict = run_inference_for_single_image(image_np, detection_graph)
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=8)
  
  # Return found objects
  # print([category_index.get(i) for i in output_dict['detection_classes'][0]])
  print(output_dict['detection_boxes'].shape)
  print(output_dict['num_detections'])

  # save image detection
  cv2.imwrite("../../outputs/image{}_detection_linux.jpg".format(counter), image_np)

  counter += 1

# show detected images
print("[INFO] show detections")
TEST_OUTPUT_IMAGE_PATHS = [os.path.join("../../outputs/", 'image{}_detection_linux.jpg'.format(i)) for i in range(1, 3)]
for image_path in TEST_OUTPUT_IMAGE_PATHS:
  print(image_path)
  image = cv2.imread(image_path)
  cv2.imshow("Detection", image)
  key = cv2.waitKey(0) & 0xFF
  if key == ord("q"):
    break

cv2.destroyAllWindows()
