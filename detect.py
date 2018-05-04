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
from multiprocessing import Process
from multiprocessing import Queue
import threading
import gc

sys.path.append('..')

#set variables
MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90
ip = '192.168.20.144:1935'
username = 'admin'
password = 'EYE3inapp'
inputQueue1 = []
inputQueue2 = []
outputQueue = Queue()
last_time_processed_frame = time.time()
last_time_frame = time.time()
last_time_thread = time.time()
start_flag = 0


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

def detect_object(sess , t = None):
	while True:
		smallest = time.time()
		pos , i = 0 , 0
		if len(inputQueue2) > 0 :
			for item in inputQueue2:
				if item < smallest:
					pos = i
				i += 1
			image_np = inputQueue1[pos]
			inputQueue1.pop(pos)
			inputQueue2.pop(pos)
			image_np_expanded = np.expand_dims(image_np, axis=0)
			(boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],feed_dict={image_tensor: image_np_expanded})
			vis_util.visualize_boxes_and_labels_on_image_array(image_np,np.squeeze(boxes),np.squeeze(classes).astype(np.int32),np.squeeze(scores),category_index,use_normalized_coordinates=True,line_thickness=8)
			for i, item in enumerate(boxes[0]):
				if scores[0][i] > 0.5 and classes[0][i] == 1:
					pass
					#print 'Person Detected'
			#if t != None:
			#	t.join()
			outputQueue.put(image_np)

			



def read_frame():
	while start_flag == 0 :
		pass
	while True:
		ret, image_np = camera.read()
		cv2.resize(image_np,(640,480))
		inputQueue1.append(image_np)
		inputQueue2.append(time.time())


		
	
		

with detection_graph.as_default():
	with tf.Session(graph=detection_graph) as sess:
		image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
		detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
		detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
		detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
		num_detections = detection_graph.get_tensor_by_name('num_detections:0')
		

		t1 = threading.Thread(target = detect_object, args=(sess,))
		'''
		t2 = threading.Thread(target = detect_object, args=(sess,t1))
		t3 = threading.Thread(target = detect_object, args=(sess,t2))
		t4 = threading.Thread(target = detect_object, args=(sess,t3))
		t5 = threading.Thread(target = detect_object, args=(sess,t4))
		t6 = threading.Thread(target = detect_object, args=(sess,t5))
		t7 = threading.Thread(target = detect_object, args=(sess,t6))
		t8 = threading.Thread(target = detect_object, args=(sess,t7))
		t9 = threading.Thread(target = detect_object, args=(sess,t8))
		t10 = threading.Thread(target = detect_object, args=(sess,t9))
		t11 = threading.Thread(target = detect_object, args=(sess,t10))
		t12 = threading.Thread(target = detect_object, args=(sess,t11))
		t13 = threading.Thread(target = detect_object, args=(sess,t12))
		t14 = threading.Thread(target = detect_object, args=(sess,t13))
		t15 = threading.Thread(target = detect_object, args=(sess,t14))
		t16 = threading.Thread(target = detect_object, args=(sess,t15))
		t17 = threading.Thread(target = detect_object, args=(sess,t16))
		t18 = threading.Thread(target = detect_object, args=(sess,t17))
		t19 = threading.Thread(target = detect_object, args=(sess,t18))
		t20 = threading.Thread(target = detect_object, args=(sess,t19))
		t21 = threading.Thread(target = detect_object, args=(sess,t20))
		'''


		t = threading.Thread(target = read_frame, args = ())
		t.daemon = True
		t.start()
		t1.start()
		'''
		t2.start()
		t3.start()
		t4.start()
		t5.start()
		t6.start()
		t7.start()
		t8.start()
		t9.start()
		t10.start()
		t11.start()
		t12.start()
		t13.start()
		t14.start()
		t15.start()
		t16.start()
		t17.start()
		t18.start()
		t19.start()
		t20.start()
		t21.start()
		'''
		start_flag = 1
		while True:
			if not outputQueue.empty():
				last_time_processed_frame = time.time()
				image_np = outputQueue.get()
				cv2.imshow('Video_Stream', cv2.resize(image_np,(640,480)))
				if cv2.waitKey(1) & 0xff == ord('q'):
					break 
				#print 'Processed Frame Time - ',time.time() - last_time_processed_frame
				last_time_processed_frame = time.time()
			'''if not inputQueue1.empty():
				img = inputQueue1.get()
				cv2.imshow('Original Stream', cv2.resize(img,(640,480)))
				if cv2.waitKey(1) & 0xff == ord('q'):
					break '''

				
cv2.destroyAllWindows()


