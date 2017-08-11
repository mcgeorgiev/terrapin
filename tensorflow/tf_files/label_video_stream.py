import os, sys

import tensorflow as tf

import numpy as np
import cv2

cap = cv2.VideoCapture(0)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Unpersists graph from file
with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

# change this as you see fit
image_paths = [sys.argv[1]]

image_data = []
for path in image_paths:
    # Read in the image_data
    with open(path, 'rb') as f:
        data = f.read()

        image_data.append(data)


# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("retrained_labels.txt")]

with tf.Session() as sess:

    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    for image in image_data:
        predictions = sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': np.array(image)})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (score = %.5f)' % (human_string, score))
