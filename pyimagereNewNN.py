#call in my vm with /home/andre/Desktop/CICD_Tutorial_Python/venv/bin/python3 pyimagere.py
#call in senecavm /home/student/catenv2/bin/python3 pyimagere.py

# --------------------------------------------------------------------------------------------------
# IMAGERE: https://github.com/imagere/neural_network.git
# DATE: 08-MAR-2020
# The objective of this code is to use a newly trained NN for image recognition
# --------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------
# ATTENTION 1: If you getting the message:
# "Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA"
# https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u
# --------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------
# 1. LOAD MODULES
# ---------------------------------------------------------------------
# check ATTENTION 1 above. Disables the warning, doesn't enable AVX/FMA
import os
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# import keras
import keras

#import miscellaneous modules
import sys
#import matplotlib.pyplot as plt #uncomment this line to show the picture. 
import numpy as np

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

# import inception_v3, these lines must be under the the import tensorflow as tf line
# ATTENTION: there is a false positive 'E0611: no name python in module tensorflow', just ignore
from tensorflow.python.keras.preprocessing import image  
from tensorflow.python.keras.applications.inception_v3 import *

# import the list of categories from the file categories.py
from newCategories import labels_to_names
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# 2. FUNCTION TO GET TENSORFLOW SESSION
# ----------------------------------------------------------------------------
def get_session():
    #config = tf.ConfigProto() # Old TF version
    config = tf.compat.v1.ConfigProto()

    # ATTENTION: UNCOMMENT THE LINE BELLOW FOR GPU !!!!
    #config.gpu_options.allow_growth = true
    # return tf.Session(config=config) # Old TF version
    return tf.compat.v1.Session(config=config)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# 3. ATTENTION: uncomment this line to select CUDA GPU to use, 
# No GPU on MATRIX!!!! Keep commented.
# ----------------------------------------------------------------------------
# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# ----------------------------------------------------------------------------
# Set the modified tf session as backend in keras
tf.compat.v1.keras.backend.set_session(get_session())
#keras.backend.tensorflow_backend.set_session(get_session())
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# 4. DEFINE NEW NN DATASET FOR USE
# ----------------------------------------------------------------------------
# Download the file from:
# https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels.h5
#model_path = os.path.join('inception_v3_weights_tf_dim_ordering_tf_kernels.h5')
model_path = os.path.join('model9Adam99.h5')

#model = InceptionV3(weights='imagenet')
model = tf.compat.v1.keras.models.load_model(model_path, custom_objects=None, compile=True)

#-----------------------------------------------------------------------------------------
class outputUnit:
  def __init__(self, pLabel, pPercentage):
    self.label = pLabel
    self.percentage = pPercentage
#-----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# GET BEST VALUES FROM OUTPUT ARRAY
# ----------------------------------------------------------------------------------------
def getBestValuesIndex(y):

    lY = sorted(y, reverse = True)[:]  # order array 
    indexes = []

    for j in range(len(lY)):
        for i in range(len(y)):
            print('aaa')
            # if (lY[j] == y[i]):
            #     indexes.append(i)

    return indexes

# ----------------------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------------------
if __name__== "__main__":
    #picPath = '/home/swarr/win2020/PRJ666/backend/'
    picPath = ''
    imageName = sys.argv[1]

    # load image
    img = image.load_img(picPath + imageName, target_size=(299, 299)) #299, 299

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    y = model.predict(x) #  predict(x, batch_size=None, verbose=0, steps=None)

    print (y[0])

    outputList = []
    count = 0

    for outPerc in y[0]:
        p = outPerc.item() # convert numpy float to python
        out = outputUnit(labels_to_names[count], round(p * 100,2))
        print (count)
        outputList.append(out)
        count+=1

    sortedList = sorted(outputList, key=lambda x: x.percentage, reverse=True)
    finalList = sortedList[:5] # get the last 5 objects (the 5 highest results)

    # for res in outputList:
    #     p = res.percentage 
    #     print (res.label +  ' ' + str(res.percentage) )


    output = {
        "nnResult":[]
    }
    for index, res in enumerate(finalList):
        label = '{}'.format(res.label)
        percentile = '{:.1f}'.format(res.percentage)
        output["nnResult"].append(
            {"label":label,
            "percentile":percentile}) #res[1] category, res[2] probability

    print(json.dumps(output))