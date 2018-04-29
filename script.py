from __future__ import division
path='train'
import numpy as np
import pandas as pd
import os
os.walk(path)
dirs=[x[0] for x in os.walk(path)]
print(dirs)
import glob
import scipy
import scipy.misc
counter=0
for dir in dirs:
   for filename in glob.iglob(dir+'/*.*'):
       img=scipy.misc.imread(filename)

       if img.shape[0]>128 and img.shape[1]>128 and img.shape[2]==3:

          counter+=1
          print(counter)
train=np.empty(shape=(counter,128,128,3))
counter=0
y=[]
for dir in dirs:
   for filename in glob.iglob(dir+'/*.*'):

       img=scipy.misc.imread(filename)
       if img.shape[0]>128 and img.shape[1]>128 and img.shape[2]==3:
          img=scipy.misc.imresize(img,(128,128))

          train[counter]=img
          counter+=1
          print(counter)
          f=filename.split('/')
          y.append(f[-2])


y=np.array(y)
train=train/255.

import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input

from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

import keras.backend as K
K.set_image_data_format('channels_last')
def PlantModel(input_shape):


   # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
   X_input = Input(input_shape)

   # Zero-Padding: pads the border of X_input with zeroes
   X = ZeroPadding2D((3, 3))(X_input)

   # layer group1 32*32*32
   # CONV -> BN -> RELU Block applied to X
   X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv1')(X)
   X = BatchNormalization(axis = 3, name = 'bn1')(X)
   X = Activation('relu')(X)
   X = Dropout(0.1)(X)
   X = MaxPooling2D((2, 2), name='max_pool1')(X)

   #layer group2 16*16*64
   X = ZeroPadding2D((2, 2))(X)
   # CONV -> BN -> RELU Block applied to X
   X = Conv2D(64, (5, 5), strides = (1, 1), name = 'conv2')(X)
   X = BatchNormalization(axis = 3, name = 'bn2')(X)
   X = Activation('relu')(X)

   X = MaxPooling2D((2, 2), name='max_pool2')(X)

   #layer group3 8*8*128
   X = ZeroPadding2D((1, 1))(X)
   # CONV -> BN -> RELU Block applied to X
   X = Conv2D(128, (3, 3), strides = (1, 1), name = 'conv3')(X)
   X = BatchNormalization(axis = 3, name = 'bn3')(X)
   X = Activation('relu')(X)

   X = MaxPooling2D((2, 2), name='max_pool3')(X)

   #layer group4 8*8*64
   # CONV -> BN -> RELU Block applied to X
   X = Conv2D(64, (1, 1), strides = (1, 1), name = 'conv4')(X)
   X = BatchNormalization(axis = 3, name = 'bn4')(X)
   X = Activation('relu')(X)


   #layer group5 4*4*32
   X = ZeroPadding2D((1, 1))(X)
   # CONV -> BN -> RELU Block applied to X
   X = Conv2D(32, (3, 3), strides = (1, 1), name = 'conv5')(X)
   X = BatchNormalization(axis = 3, name = 'bn5')(X)
   X = Activation('relu')(X)



   X = MaxPooling2D((2, 2), name='max_pool5')(X)

   # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
   X = Flatten()(X)
   X = Dense(128, activation='sigmoid', name='fc1')(X)
   X = Dense(32, activation='sigmoid', name='fc2')(X)
   X = Dense(12, activation='sigmoid', name='fc3')(X)


   model = Model(inputs = X_input, outputs = X, name='HappyModel')

   ### END CODE HERE ###

   return model


plantModel = PlantModel(((128,128,3)))
plantModel.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

from sklearn.cross_validation import  train_test_split
from keras.utils.np_utils import to_categorical
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
y=le.fit_transform(y)
temp=y
y=to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.33, random_state=42)
plantModel.fit(x=X_train, y=y_train, epochs=20, batch_size=35)
plantModel.evaluate(X_test,y_test)
########################################
test_path='test'
counter=0
temp_test = pd.DataFrame({'file' : [np.nan],'prediction':np.nan})
for filename in glob.iglob(test_path+'/*.*'):
   img=scipy.misc.imread(filename)
   data = pd.DataFrame({"file": filename.split('/')[-1],"prediction":'x',},index=[counter])
   temp_test=temp_test.append(data)
   counter+=1
   print(filename)

test=np.empty(shape=(counter,128,128,3))

counter=0

for filename in glob.iglob(test_path+'/*.*'):
   img=scipy.misc.imread(filename)
   img=scipy.misc.imresize(img,(128,128,3))


   test[counter]=img
   counter+=1
   print(counter)
test=test/255.
temp_test=temp_test.iloc[1:]
np.set_printoptions(suppress=True)


pred=happyModel.predict(test)
#pred=np.round(pred)
sub_result=[]
for i in range(counter):
   cat=le.inverse_transform(np.argmax(pred[i]))
   sub_result.append(cat)
sub_result=np.array(sub_result)
temp_test['species']=sub_result

submit=pd.read_csv('sample_submission.csv')
result=pd.merge(submit,temp_test,on='file')
result['species']=result['species_y']
result=result.drop(['species_x','prediction','species_y'],axis=1)
result.to_csv('client_predictions.csv',index=False)
