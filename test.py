#!/usr/bin/env python
# coding: utf-8

# In[60]:


from generator import generator
from valid_generator import valid_generator
#from __future__ import print_function
import json
import os.path
import random as ra
import numpy as np
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Input, Dense, Dropout, BatchNormalization, Reshape, Lambda, Embedding, LSTM, Conv2D, MaxPooling2D, TimeDistributed, RepeatVector, Concatenate
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing import sequence
from scipy import ndimage, misc
import itertools 
import pickle
import tensorflow as tf


# In[114]:


path = 'CLEVR_v1.0'
questions_path = path + '/questions/CLEVR_' + 'val' + '_questions.json'
with open(questions_path) as f:
    data_test = json.load(f)


def new_xtext(x_text):
	tokenizer = Tokenizer(num_words=vocab_size)
	tokenizer.fit_on_texts(x_text)
	sequences = tokenizer.texts_to_sequences(x_text)
	x_text = sequence.pad_sequences(sequences, maxlen=sequence_length)
	return x_text

def get_label(data_real):
	labels = {}
	num_labels = 0
	y = []
	x_text=[]
	for q in data_real:
		if not q['answer'] in labels:
			labels[q['answer']] = num_labels
			num_labels += 1
		y.append(labels[q['answer']])
		x_text.append(q['question'])
	return x_text,y,num_labels,labels	

####################   MODEL AA GAYA     ###########################

def process_image(x):
    target_height, target_width = 128, 128
    rotation_range = .05  # In radians
    degs = ra.uniform(-rotation_range, rotation_range)

    x = tf.image.resize_images(x, (target_height, target_width), method=tf.image.ResizeMethod.AREA)
    x = tf.contrib.image.rotate(x, degs)

    return x

def get_relation_vectors(x):
    objects = []
    relations = []
    shape = K.int_shape(x)
    k = 25     # Hyperparameter which controls how many objects are considered
    keys = []

    while k > 0:
        i = ra.randint(0, shape[1] - 1)
        j = ra.randint(0, shape[2] - 1)

        if not (i, j) in keys:
            keys.append((i, j))
            objects.append(x[:, i, j, :])
            k -= 1

    for i in range(len(objects)):
        for j in range(i, len(objects)):
            relations.append(K.concatenate([objects[i], objects[j]], axis=1))
    return K.permute_dimensions(K.stack([r for r in relations], axis=0), [1, 0, 2])


# In[74]:


samples = 699968
epochs = 25
batch_size = 32
valid_batch_size = 16
learning_rate = .00025
vocab_size = 1024
sequence_length = 64
img_rows, img_cols = 320, 480
num_labels=28
image_input_shape = (img_rows, img_cols, 3)

def relational_model():
	image_input_shape = (img_rows, img_cols, 3)

	text_inputs = Input(shape=(sequence_length,), name='text_input')
	text_x = Embedding(vocab_size, 128)(text_inputs)
	text_x = LSTM(128)(text_x)

	image_inputs = Input(shape=image_input_shape, name='image_input')
	image_x = Lambda(process_image)(image_inputs)
	image_x = Conv2D(24, kernel_size=(3, 3), strides=2, activation='relu')(image_inputs)
	image_x = BatchNormalization()(image_x)
	image_x = Conv2D(24, kernel_size=(3, 3), strides=2, activation='relu')(image_x)
	image_x = BatchNormalization()(image_x)
	image_x = Conv2D(24, kernel_size=(3, 3), strides=2, activation='relu')(image_x)
	image_x = BatchNormalization()(image_x)
	image_x = Conv2D(24, kernel_size=(3, 3), strides=2, activation='relu')(image_x)
	image_x = BatchNormalization()(image_x)
	shape = K.int_shape(image_x)

	RN_inputs = Input(shape=(1, (2 * shape[3]) + K.int_shape(text_x)[1]))
	RN_x = Dense(256, activation='relu')(RN_inputs)	
	RN_x = Dense(256, activation='relu')(RN_x)
	RN_x = Dense(256, activation='relu')(RN_x)
	RN_x = Dropout(.5)(RN_x)
	RN_outputs = Dense(256, activation='relu')(RN_x)
	RN = Model(inputs=RN_inputs, outputs=RN_outputs)

	relations = Lambda(get_relation_vectors)(image_x)           # Get tensor [batch, relation_ID, relation_vectors]
	question = RepeatVector(K.int_shape(relations)[1])(text_x)  # Shape question vector to same size as relations
	relations = Concatenate(axis=2)([relations, question])      # Merge tensors [batch, relation_ID, relation_vectors, question_vector]
	g = TimeDistributed(RN)(relations)                          # TimeDistributed applies RN to relation vectors.
	g = Lambda(lambda x: K.sum(x, axis=1))(g)                   # Sum over relation_ID

	f = Dense(256, activation='relu')(g)
	f = Dropout(.5)(f)
	f = Dense(256, activation='relu')(f)
	f = Dropout(.5)(f)
	outputs = Dense(num_labels, activation='softmax')(f)

	## Train model
	model = Model(inputs=[text_inputs, image_inputs], outputs=outputs)
	return model


# In[156]:


#from model_with_valid import relational_model
model_weight="RN_weights/Batch_size:32_spe:21874epochs:25.0.h5"

def test_RN(img,question,tokenizer,model_weight):
    model=relational_model()
    model.load_weights(model_weight)
    
    sequences = tokenizer.texts_to_sequences([ques])
    #print(sequences)
    x_text = sequence.pad_sequences(sequences, maxlen=sequence_length)
    #print(x_text)

    test_img = np.reshape(img,(1,img.shape[0],img.shape[1],img.shape[2]))
    #print(test_img.shape)
    
    out = model.predict([x_text,test_img])
    return out


# In[158]:


with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    
with open('labels.pickle', 'rb') as handle:
    labels = pickle.load(handle)


for i in range(0,4):
	data = data_test['questions'][i]
	image_file = 'CLEVR_v1.0/images/val/'+ data['image_filename']

	img = misc.imread(image_file, mode='RGB')

	ques = data['question']
	ans=data['answer']
	print(image_file)
	print(ques)
	print(ans)
	out=test_RN(img,ques,tokenizer,model_weight)

	#print(out)
	print(np.argmax(out))

print(labels)





