# https://github.com/keras-team/keras/issues/8418

# ______________________________________________________________________________
# Imports

import argparse
import keras
import os

import tensorflow as tf
import numpy      as np

from keras.models       import Model
from keras.applications import InceptionV3
from keras.models       import Sequential
from keras.layers       import Dense, Dropout, Activation
from keras.optimizers   import SGD

# ______________________________________________________________________________
# Constants.

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = 'downloads' # This directory stores subdirectories that are image class titles, like 'cat'. 
					  # Those subdirectories contain the training images for that class.
INCEPTIONV3 = InceptionV3()

INPUT_HEIGHT = 299 #224
INPUT_WIDTH  = 299 #224
INPUT_MEAN   = 127.5
INPUT_STD    = 127.5

# ______________________________________________________________________________
# Preprocess image file function.

def read_file(file_name):
	"""
	Convert string of .jpg file path to normalized np array for image processing.
	"""
	print ('----------------- ' + file_name)
	result = ""
	with tf.compat.v1.Session() as ses:
		file_reader   = tf.io.read_file(file_name, "file_reader")
		image_reader  = tf.image.decode_jpeg(file_reader, channels = 3, name='jpeg_reader')
		float_caster  = tf.cast(image_reader, tf.float32)
		dims_expander = tf.expand_dims(float_caster, 0)
		resized       = tf.compat.v1.image.resize_bilinear(dims_expander, [INPUT_HEIGHT, INPUT_WIDTH])
		normalized    = tf.divide(tf.subtract(resized, [INPUT_MEAN]), [INPUT_STD])
	
	#sess   = tf.compat.v1.Session()
		result = ses.run(normalized)
	
	return result

def build_tl_model(original_model):
	"""
	Perform 'surgery' on a pretrained model. Then add layers to create a new model
	that has just one final, trainable layer with softmax activation for 
	multi-class output.
	"""
	# ______________________________________________________________________________
	# Extract needed info from pre-trained model.
	bottleneck_input  = original_model.get_layer(index=0).input
	bottleneck_output = original_model.get_layer(index=-2).output
	bottleneck_model = Model(inputs=bottleneck_input, outputs=bottleneck_output)

	# ______________________________________________________________________________
	# Freeze these layers so we are not retraining the full model. 
	for layer in bottleneck_model.layers:
		layer.trainable = False

	# ______________________________________________________________________________
	# Build new transfer learning model.
	new_model = Sequential()
	new_model.add(bottleneck_model)
	if args.verbose:
		print('Model summary before final layer addition:', new_model.summary())
	
	NUM_CLASSES = len(os.listdir(IMG_DIR))  # How many image classes are in our new data?
	BOTTLENECK_DIM = original_model.get_layer(index=-2).output.shape.dims[1]  # The number of nodes in the second to last layer of the pre-trained model.

	new_model.add(Dense(NUM_CLASSES, 
						input_dim  = 2048,  # BOTTLENECK_DIM
						activation = 'softmax'))  # Convert outputs to probabilities.
	if args.verbose:
		print('Model summary after final layer addition:', new_model.summary())
		new_model.save('my_model.h5')

	return new_model

# ______________________________________________________________________________
# Process data for transfer learning.

def process_data():
	"""
	Convert filepaths to list of np pixel arrays in proper format.

	In this case, `processed_imgs_array` is array of size:

	(n, 224, 224, 3)

	where n = total number training images.
	"""
	processed_imgs_list = []
	labels = []
	for i, subdir in enumerate(os.listdir(IMG_DIR)):
		for file_name in os.listdir(IMG_DIR+'/'+subdir):
			#print ('--------------' + BASE_DIR + '/' + IMG_DIR + '/' + subdir + '/' + file_name)
			processed_imgs_list.append(read_file(BASE_DIR + '/' + IMG_DIR + '/' + subdir + '/' + file_name))
			labels.append(i)

	ttl_num_new_imgs = len(processed_imgs_list)
	#processed_imgs_array = np.asarray(processed_imgs_list).reshape(ttl_num_new_imgs,224,224,3)
	processed_imgs_array = np.asarray(processed_imgs_list).reshape(ttl_num_new_imgs,299,299,3)

	return processed_imgs_array, labels, processed_imgs_list

# ______________________________________________________________________________
# Retrain.

def retrain(new_model, processed_imgs_array, labels):
	# For a binary classification problem
	new_model.compile(optimizer = 'rmsprop',
	                  loss      = 'binary_crossentropy',
	                  metrics   = ['accuracy'])

	one_hot_labels = keras.utils.to_categorical(labels, num_classes=2)

	new_model.fit(processed_imgs_array, one_hot_labels, epochs=2, batch_size=20)
	
	return new_model

if __name__ == '__main__':
	# Command line options: 
	parser =  argparse.ArgumentParser()
	parser.add_argument('--verbose', action='store_true')
	args = parser.parse_args()

	new_model = build_tl_model(INCEPTIONV3)
	processed_imgs_array, labels, processed_imgs_list = process_data()
	new_model = retrain(new_model, processed_imgs_array, labels)
	if args.verbose:
		print(new_model.predict(processed_imgs_list[0].reshape(1,299,299,3)))  # Newly trained model prediction. [0,1] = [cat, dog].
		print(INCEPTIONV3.predict(processed_imgs_list[0].reshape(1,299,299,3)).argmax(-1))  # Original Inception model prediction.

	