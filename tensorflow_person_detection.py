import cv2
import numpy as np
import sys
import os
import time
import tensorflow as tf
from utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util
from collections import defaultdict

sys.path.append('..')

#set variables
MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90
ip = '192.168.20.144:1935'
username = 'admin'
password = 'EYE3inapp'
last_time = time.time()
 

#load graph
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

#load label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


#capture video stream from ip camera
camera = cv2.VideoCapture('rtmp://'+ ip + '/flash/12:'+ username + ':' + password)



with detection_graph.as_default():
	with tf.Session(graph=detection_graph) as sess:
		image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
		detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
		detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
		detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
		num_detections = detection_graph.get_tensor_by_name('num_detections:0')
		while True:
			#read frame 
			ret, image_np = camera.read()
			#resize image for easier manipulation
			cv2.resize(image_np,(640,480))
			image_np_expanded = np.expand_dims(image_np, axis=0)
			(boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],feed_dict={image_tensor: image_np_expanded})
			for i, item in enumerate(boxes[0]):
				#class id = 1 for person and accuracy should be greater than 0.5 (50%)
				if scores[0][i] > 0.5 and classes[0][i] == 1:
					#only if person is detected, draw bounding boxes
					vis_util.visualize_boxes_and_labels_on_image_array(image_np,np.squeeze(boxes),np.squeeze(classes).astype(np.int32),np.squeeze(scores),category_index,use_normalized_coordinates=True,line_thickness=8)
					#to find the number of frames per second
					print time.time() - last_time
					last_time = time.time()
			#display 
			cv2.imshow('Video_Stream', cv2.resize(image_np,(640,480)))
			if cv2.waitKey(1) & 0xff == ord('q'):
				break

cv2.destroyAllWindows()

