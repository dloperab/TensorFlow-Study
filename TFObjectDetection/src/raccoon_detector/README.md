# TensorFlow - TensorFlow Object Detection API

## Raccoon Detector

### Pipeline to reproduce:

1. Images for training, test y evalidation:
    * [Training/Test Images](images/).
    * [Evaluation Images](images/eval/).
2. Annotations generated in Pascal VOC format with [labelImg tool](https://github.com/tzutalin/labelImg). The annotations are in /images folder.
3. Convert xml files (annotations) to csv format.
    * *python scripts/preprocessing/xml_to_csv.py -i /images -o /annotations/raccoon_labels.csv*
4. Split labels for training (80%) and testing (20%).
    * *python scripts/preprocessing/split_labels.py -i /annotations/raccoon_labels.csv -o /annotations*
5. Generate [TFRecord](https://www.tensorflow.org/api_docs/python/tf/io#tfrecords_format_details) files for training and testing.
    * *python scripts/preprocessing/generate_tfrecord.py --csv_input=/annotations/train_labels.csv --output_path=/annotations/train.record --img_path=/datasets/images*
    * *python scripts/preprocessing/generate_tfrecord.py --csv_input=/annotations/test_labels.csv --output_path=/annotations/test.record --img_path=/images*
6. Copy the pre-trained model to use from [TensorFlow Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) to folder /pre-trained-model. For this example was used the "ssd_inception_v2_coco" model.
7. Generate .pbtxt file in folder /annotations/label_map.pbtxt.
8. Copy the config file of the used model to folder /training/ssd_inception_v2_coco.config and modify all the needed parameters.
9. Customize "train.py" script copied from "TF/models/research/object_detection/legacy/train.py". Copy this file to main folder and execute using:
    * *python train.py --logtostderr --train_dir=training_output/ --pipeline_config_path=training/ssd_inception_v2_coco.config*
10. Monitor the training using TensorBoard.
    * *tensorboard --logdir=training_output/*
11. Once the training job is complete, export a trained inference graph.
    * *python scripts/export_inference_graph.py --input_type image_tensor --pipeline_config_path training/ssd_inception_v2_coco.config --trained_checkpoint_prefix training/model.ckpt-2000 --output_directory trained-inference-graphs/inference_graph_v1.pb*
12. Evaluate trained model with eval images.
    * *python eval.py*

For more details:
* [How to train your own Object Detector with TensorFlow’s Object Detector API](https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9).
* [Training Custom Object Detector](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html).
* [Building a Toy Detector with Tensorflow Object Detection API](https://towardsdatascience.com/building-a-toy-detector-with-tensorflow-object-detection-api-63c0fdf2ac95).