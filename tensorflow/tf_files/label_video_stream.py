import os, sys, time

import tensorflow as tf

import numpy as np
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Unpersists graph from file
with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("retrained_labels.txt")]


cap = cv2.VideoCapture('Untitled.avi')

with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    while(cap.isOpened()):
        time.sleep(.3)
        print time.time()
        ret, frame = cap.read()
        image = np.array(frame)#.reshape(1,720,1280,3)
        predictions = sess.run(softmax_tensor, \
                 {'DecodeJpeg:0': image})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            if score > 0.5:
                print('%s (score = %.5f)' % (human_string, score))

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
