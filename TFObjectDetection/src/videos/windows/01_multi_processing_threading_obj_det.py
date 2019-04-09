import sys
import os
sys.path.append(os.environ['TF_RESEARCH'])

import numpy as np
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
import cv2
import multiprocessing

from imutils.video import WebcamVideoStream
from imutils.video import FPS
from multiprocessing import Queue, Pool
from queue import PriorityQueue

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# What model to download
# Models can bee found here: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28'

# construct paths
# base path where we will save our models
PATH_TO_OBJ_DETECTION = 'C:/tensorflow/models/research/object_detection'
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
        min_score_thresh=.6,
        use_normalized_coordinates=True,
        line_thickness=4)

    return image_np     

def worker(input_q, output_q):
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    fps = FPS().start()
    while True:
        fps.update()
        frame = input_q.get()

        # Check frame object is a 2-D array (video) or 1-D (webcam)
        if len(frame) == 2:
            frame_rgb = cv2.cvtColor(frame[1], cv2.COLOR_BGR2RGB)
            output_q.put((frame[0], detect_objects(frame_rgb, sess, detection_graph)))
        else:
            frame_rgb = cv2.cvtColor(frame[1], cv2.COLOR_BGR2RGB)
            output_q.put(detect_objects(frame_rgb, sess, detection_graph))

    fps.stop()
    sess.close()

if __name__ == '__main__':
    download_model()

    # args
    queue_size = 5 # Size of the queue
    num_workers = 2 # Number of workers
    input_videos = "../../data/videos/traffic.mp4"
    display = 0 # Whether or not frames should be displayed
    output = 1 # Whether or not modified videos shall be writen

    # Multiprocessing: Init input and output Queue, output Priority Queue and pool of workers
    input_q = Queue(maxsize=queue_size)
    output_q = Queue(maxsize=queue_size)
    output_pq = PriorityQueue(maxsize=3*queue_size)
    pool = Pool(num_workers, worker, (input_q,output_q))

    # Define the video stream
    # created a threaded video stream and start the FPS counter
    vs = cv2.VideoCapture(input_videos)
    fps = FPS().start()

    # Define the codec and create VideoWriter object
    if output:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('../../../outputs/video_detection.avi',
                              fourcc, vs.get(cv2.CAP_PROP_FPS),
                              (int(vs.get(cv2.CAP_PROP_FRAME_WIDTH)),
                               int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # Start reading and treating the video stream
    if display > 0:
        print()
        print("=====================================================================")
        print("Starting video acquisition. Press 'q' (on the video windows) to stop.")
        print("=====================================================================")
        print()
    
    countReadFrame = 0
    countWriteFrame = 1
    nFrame = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    firstReadFrame = True
    firstTreatedFrame = True
    firstUsedFrame = True
    while True:
        # Check input queue is not full
        if not input_q.full():
            # Read frame and store in input queue
            ret, frame = vs.read()
            if ret:            
                input_q.put((int(vs.get(cv2.CAP_PROP_POS_FRAMES)),frame))
                countReadFrame = countReadFrame + 1
                if firstReadFrame:
                    print(" --> Reading first frames from input file. Feeding input queue.\n")
                    firstReadFrame = False

        # Check output queue is not empty
        if not output_q.empty():
            # Recover treated frame in output queue and feed priority queue
            output_pq.put(output_q.get())
            if firstTreatedFrame:
                print(" --> Recovering the first treated frame.\n")
                firstTreatedFrame = False

        # Check output priority queue is not empty
        if not output_pq.empty():
            prior, output_frame = output_pq.get()
            if prior > countWriteFrame:
                output_pq.put((prior, output_frame))
            else:
                countWriteFrame = countWriteFrame + 1
                output_rgb = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)

                # Write the frame in file
                if output:
                    out.write(output_rgb)

                # Display the resulting frame
                if display:
                    cv2.imshow('frame', output_rgb)
                    fps.update()

                if firstUsedFrame:
                    print(" --> Start using recovered frame (displaying and/or writing).\n")
                    firstUsedFrame = False

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        print("Read frames: %-3i %% -- Write frame: %-3i %%" % (int(countReadFrame/nFrame * 100), int(countWriteFrame/nFrame * 100)), end ='\r')
        if((not ret) & input_q.empty() & output_q.empty() & output_pq.empty()):
            break

    print("\nFile have been successfully read and treated:\n  --> {}/{} read frames \n  --> {}/{} write frames \n".format(countReadFrame,nFrame,countWriteFrame-1,nFrame))

    # When everything done, release the capture
    print("[INFO] releasing objects...")
    fps.stop()
    pool.terminate()
    vs.release()
    if output:
        out.release()
    cv2.destroyAllWindows()
    print("[INFO] objects released...")
