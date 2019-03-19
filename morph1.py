import keras
from keras.layers import Input, Embedding, LSTM, Dense
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten , Input, BatchNormalization, Activation
from keras.models import Model, load_model
from keras.utils import to_categorical
from keras.utils import plot_model

#import pydot_ng as pydot


import os
import tensorflow as tf
import numpy as np
import h5py
from PIL import Image
from PIL import ImageStat
from scipy.stats import kurtosis, skew

path_img = 'originals/'

train_images = []
test_images = []
count_img = 0

for f in os.listdir(path_img):
	count_img += 1
	img = Image.open(path_img+f)
	stat = ImageStat.Stat(img)
	data = []

	data.append(stat.mean)

	data.append(stat.median)

	data.append(stat.var)
    
  
	data.append([np.mean(kurtosis(img)[:,0]),np.mean(kurtosis(img)[:,1]),np.mean(kurtosis(img)[:,2])])

	data.append([np.mean(skew(img)[:,0]),np.mean(skew(img)[:,1]),np.mean(skew(img)[:,2])])

	# data = np.asarray(img, dtype="uint8")
	if count_img < 1382:
		train_images.append(data)

	else:
		test_images.append(data)

print(data)

train_data = np.array(train_images)
# train_images = train_images / 255
# trai_dim0,trai_dim1,trai_dim2 = train_images.shape[0], train_images.shape[2], train_images.shape[1]
# train_data = train_images.reshape(trai_dim0,trai_dim1,trai_dim2, 1)

test_data = np.array(test_images)
# test_images = test_images / 255
# test_data = test_images.reshape(300,trai_dim1,trai_dim2, 1)

#print(train_images.shape[0],train_images.shape[2],train_images.shape[1])

file_dataset= open("dataset_param_orig_bgs_tri.txt", "r")

train_rad_e = []
train_it_e  = []
train_rad_d = []
train_it_d  = []

test_rad_e  = []
test_it_e   = []
test_rad_d  = []
test_it_d   = []

count_dtset = 0

for line in file_dataset:
	count_dtset += 1
	x1,x2,x3,x4 , xxx , yyy =line.split(";")
	if count_dtset<1382:

		train_rad_e.append(x1)
		train_it_e.append(x2)
		train_rad_d.append(x3)
		train_it_d.append(x4)
	else:
		test_rad_e.append(x1)
		test_it_e.append(x2)
		test_rad_d.append(x3)
		test_it_d.append(x4)

train_rad_e = np.array(train_rad_e)
train_it_e = np.array(train_it_e)
train_rad_d = np.array(train_rad_d)
train_it_d = np.array(train_it_d)


cat_train_rad_e = []
for x in range(len(train_rad_e)):
	if train_rad_e[x]=="3":
		cat_train_rad_e.append(0)
	if train_rad_e[x]=="5":
		cat_train_rad_e.append(1)
	if train_rad_e[x]=="7":
		cat_train_rad_e.append(2)
	if train_rad_e[x]=="9":
		cat_train_rad_e.append(3)
	if train_rad_e[x]=="11":
		cat_train_rad_e.append(4)
	if train_rad_e[x]=="13":
		cat_train_rad_e.append(5)
	if train_rad_e[x]=="15":
		cat_train_rad_e.append(6)
	if train_rad_e[x]=="17":
		cat_train_rad_e.append(7)
	if train_rad_e[x]=="19":
		cat_train_rad_e.append(8)
	if train_rad_e[x]=="21":
		cat_train_rad_e.append(9)
data_train_rad_e = to_categorical(cat_train_rad_e,10)

cat_test_rad_e = []
for x in range(len(test_rad_e)):
	if test_rad_e[x]=="3":
		cat_test_rad_e.append(0)
	if test_rad_e[x]=="5":
		cat_test_rad_e.append(1)
	if test_rad_e[x]=="7":
		cat_test_rad_e.append(2)
	if test_rad_e[x]=="9":
		cat_test_rad_e.append(3)
	if test_rad_e[x]=="11":
		cat_test_rad_e.append(4)
	if test_rad_e[x]=="13":
		cat_test_rad_e.append(5)
	if test_rad_e[x]=="15":
		cat_test_rad_e.append(6)
	if test_rad_e[x]=="17":
		cat_test_rad_e.append(7)
	if test_rad_e[x]=="19":
		cat_test_rad_e.append(8)
	if test_rad_e[x]=="21":
		cat_test_rad_e.append(9)
data_test_rad_e = to_categorical(cat_test_rad_e,10)


cat_train_rad_d = []
for x in range(len(train_rad_d)):
	if train_rad_d[x]=="3":
		cat_train_rad_d.append(0)
	if train_rad_d[x]=="5":
		cat_train_rad_d.append(1)
	if train_rad_d[x]=="7":
		cat_train_rad_d.append(2)
	if train_rad_d[x]=="9":
		cat_train_rad_d.append(3)
	if train_rad_d[x]=="11":
		cat_train_rad_d.append(4)
	if train_rad_d[x]=="13":
		cat_train_rad_d.append(5)
	if train_rad_d[x]=="15":
		cat_train_rad_d.append(6)
	if train_rad_d[x]=="17":
		cat_train_rad_d.append(7)
	if train_rad_d[x]=="19":
		cat_train_rad_d.append(8)
	if train_rad_d[x]=="21":
		cat_train_rad_d.append(9)
data_train_rad_d = to_categorical(cat_train_rad_d,10)

cat_test_rad_d = []
for x in range(len(test_rad_d)):
	if test_rad_d[x]=="3":
		cat_test_rad_d.append(0)
	if test_rad_d[x]=="5":
		cat_test_rad_d.append(1)
	if test_rad_d[x]=="7":
		cat_test_rad_d.append(2)
	if test_rad_d[x]=="9":
		cat_test_rad_d.append(3)
	if test_rad_d[x]=="11":
		cat_test_rad_d.append(4)
	if test_rad_d[x]=="13":
		cat_test_rad_d.append(5)
	if test_rad_d[x]=="15":
		cat_test_rad_d.append(6)
	if test_rad_d[x]=="17":
		cat_test_rad_d.append(7)
	if test_rad_d[x]=="19":
		cat_test_rad_d.append(8)
	if test_rad_d[x]=="21":
		cat_test_rad_d.append(9)
data_test_rad_d = to_categorical(cat_test_rad_d,10)


cat_train_it_e = []
for x in range(len(train_it_e)):
	if train_it_e[x]=="0":
		cat_train_it_e.append(0)
	if train_it_e[x]=="1":
		cat_train_it_e.append(1)
	if train_it_e[x]=="2":
		cat_train_it_e.append(2)
	if train_it_e[x]=="3":
		cat_train_it_e.append(3)
data_train_it_e = to_categorical(cat_train_it_e,4)

cat_test_it_e = []
for x in range(len(test_it_e)):
	if test_it_e[x]=="0":
		cat_test_it_e.append(0)
	if test_it_e[x]=="1":
		cat_test_it_e.append(1)
	if test_it_e[x]=="2":
		cat_test_it_e.append(2)
	if test_it_e[x]=="3":
		cat_test_it_e.append(3)
data_test_it_e = to_categorical(cat_test_it_e,4)


cat_train_it_d = []
for x in range(len(train_it_d)):
	if train_it_d[x]=="0":
		cat_train_it_d.append(0)
	if train_it_d[x]=="1":
		cat_train_it_d.append(1)
	if train_it_d[x]=="2":
		cat_train_it_d.append(2)
	if train_it_d[x]=="3":
		cat_train_it_d.append(3)
data_train_it_d = to_categorical(cat_train_it_d,4)

cat_test_it_d = []
for x in range(len(test_it_d)):
	if test_it_d[x]=="0":
		cat_test_it_d.append(0)
	if test_it_d[x]=="1":
		cat_test_it_d.append(1)
	if test_it_d[x]=="2":
		cat_test_it_d.append(2)
	if test_it_d[x]=="3":
		cat_test_it_d.append(3)
data_test_it_d = to_categorical(cat_test_it_d,4)

#print(data_train_rad_e[2])

#['11' '13' '15' '17' '19' '21' '3' '7' '9']
# # Find the unique numbers from the train labels
# classes_rad_e ,count_classes_rad_e = np.unique(train_rad_e, return_counts=True)
# classes_rad_d ,count_classes_rad_d = np.unique(train_rad_d, return_counts=True)
# classes_it_e ,count_classes_it_e = np.unique(train_it_e, return_counts=True)
# classes_it_d ,count_classes_it_d = np.unique(train_it_d, return_counts=True)
# print('classes rad e ', classes_rad_e)
# # print(dict(zip(classes_rad_e ,count_classes_rad_e)))
# classes_rad_e ,count_classes_rad_e = np.unique(test_rad_e, return_counts=True)
# print('test rad e ', classes_rad_e)
# # print(dict(zip(classes_rad_e ,count_classes_rad_e)))

# print('classes rad d ', classes_rad_d)
# # print(dict(zip(classes_rad_d ,count_classes_rad_d)))
# classes_rad_d ,count_classes_rad_d = np.unique(test_rad_d, return_counts=True)
# print('test rad d ', classes_rad_d)
# # print(dict(zip(classes_rad_d ,count_classes_rad_d)))

# print('classes it e ', classes_it_e)
# # print(dict(zip(classes_it_e ,count_classes_it_e)))
# classes_it_e ,count_classes_it_e = np.unique(test_it_e, return_counts=True)
# print('test it e ', classes_it_e)
# # print(dict(zip(classes_it_e ,count_classes_it_e)))

# print('classes it d ', classes_it_d)
# # print(dict(zip(classes_it_d ,count_classes_it_d)))

# classes_it_d ,count_classes_it_d = np.unique(test_it_d, return_counts=True)
# print('test it d ', classes_it_d)
# # print(dict(zip(classes_it_d ,count_classes_it_d)))


def createModel():

	input_layer = Input(shape=(5,3))


	layer0 = Flatten()(input_layer)
	layer1520 = Dense(200)(layer0)
	layer15_2 = BatchNormalization()(layer1520)
	layer1512 = Activation('relu')(layer15_2)
	layer1403 = Dropout(0.7)(layer1512)

	layer1520 = Dense(150)(layer1403)
	layer15_2 = BatchNormalization()(layer1520)
	layer1512 = Activation('relu')(layer15_2)
	layer1403 = Dropout(0.7)(layer1512)

	layer1520 = Dense(100)(layer1403)
	layer15_2 = BatchNormalization()(layer1520)
	layer1512 = Activation('relu')(layer15_2)
	layer1403 = Dropout(0.7)(layer1512)

	layer1520 = Dense(50)(layer1403)
	layer15_2 = BatchNormalization()(layer1520)
	layer1512 = Activation('relu')(layer15_2)
	layer1403 = Dropout(0.7)(layer1512)

	#out1
	layer1530 = Dense(20)(layer1403)
	layer15_3 = BatchNormalization()(layer1530)
	layer1513 = Activation('relu')(layer15_3)
	out1 = Dropout(0.5)(layer1513)

	#out2
	layer1530 = Dense(8)(layer1403)
	layer15_3 = BatchNormalization()(layer1530)
	layer1513 = Activation('relu')(layer15_3)
	out2 = Dropout(0.5)(layer1513)

	#out3
	layer1530 = Dense(20)(layer1403)
	layer15_3 = BatchNormalization()(layer1530)
	layer1513 = Activation('relu')(layer15_3)
	out3 = Dropout(0.5)(layer1513)

	#out4
	layer1530 = Dense(8)(layer1403)
	layer15_3 = BatchNormalization()(layer1530)
	layer1513 = Activation('relu')(layer15_3)
	out4 = Dropout(0.55)(layer1513)



	output1 = Dense(10, activation='softmax')(out1)
	output2 = Dense(4, activation='softmax')(out2)
	output3 = Dense(10, activation='softmax')(out3)
	output4 = Dense(4, activation='softmax')(out4)

	model = Model(inputs=input_layer , outputs=[output1 , output2 , output3 , output4])
    # model = Sequential()
	return model

cnn_model = createModel()

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
sess = tf.Session(config=config)
keras.backend.set_session(sess)

opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)
opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_data = np.reshape(train_data, (train_data.shape[0], 1, train_data.shape[1]))

cnn_model.fit(train_data, [data_train_rad_e,data_train_it_e,data_train_rad_d,data_train_it_d], epochs=1000, batch_size=500 ,shuffle=True, verbose = 1)

#cnn_model.fit(train_data, [data_train_rad_e,data_train_it_e,data_train_rad_d,data_train_it_d], validation_data=(test_data,[data_test_rad_e,data_test_it_e,data_test_rad_d,data_test_it_d]) ,epochs=3, batch_size=10 ,shuffle=True)


cnn_model.save('original_bignn_01.h5')
cnn_model = load_model('original_bignn_01.h5')
cnn_model.summary()

# os.environ["PATH"] += os.pathsep + 'c:/users/user/anaconda3/envs/tensorflow/lib/site-packages/'

# pydot.find_graphviz()


# plot_model(cnn_model, show_shapes=True, to_file='model.png')
test_data = np.reshape(test_data, (test_data.shape[0], 1, test_data.shape[1]))
score = cnn_model.evaluate(test_data,[data_test_rad_e,data_test_it_e,data_test_rad_d,data_test_it_d], verbose=1, batch_size = 300)
print(cnn_model.metrics_names," : ", score)
