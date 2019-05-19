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



def new_xtext(x_text):
	tokenizer = Tokenizer(num_words=vocab_size)
	tokenizer.fit_on_texts(x_text)
	sequences = tokenizer.texts_to_sequences(x_text)
	x_text = sequence.pad_sequences(sequences, maxlen=sequence_length)
	return x_text,tokenizer

def x_text_valid(x_text,tokenizer):
	sequences = tokenizer.texts_to_sequences(x_text)
	x_text = sequence.pad_sequences(sequences, maxlen=sequence_length)
	return x_text


def y_valid(data,labels):
        y = []
        x_text=[]
        for q in data:
                y.append(labels[q['answer']])
                x_text.append(q['question'])
        return x_text,y



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

####################   MODEL     ###########################

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

############# Parameter #############

validation_sample = 2000
samples = 200000
epochs = 25
batch_size = 32
valid_batch_size = 8
learning_rate = .00025
vocab_size = 1024
sequence_length = 64
img_rows, img_cols = 320, 480
image_input_shape = (img_rows, img_cols, 3)
path = 'CLEVR_v1.0'
questions_path = path + '/questions/CLEVR_' + 'train' + '_questions.json'


### save training data questions file as pickle by reading this way###
#with open(questions_path) as f:
#    data = json.load(f)
#data_real = data['questions']

with open("real_data.pickle", 'rb') as f:
    data_real = pickle.load(f)

steps_per_epoch = int(samples/batch_size)
####################### GETTING LABELS ###################
x_text,y,num_labels,labels=get_label(data_real)
x_text,tokenizer=new_xtext(x_text)

##### Saving labels #####
with open("labels.pickle","wb") as f:
        pickle.dump(labels,f,protocol=pickle.HIGHEST_PROTOCOL)


########## Saving tokenizer #############
with open("tokenizer.pickle","wb") as f:
	pickle.dump(tokenizer,f,protocol=pickle.HIGHEST_PROTOCOL)


train_gen = generator(data_real,samples,vocab_size, sequence_length,x_text,y,num_labels,batch_size)


########## Validation #########################
valid_questions_path = path + '/questions/CLEVR_' + 'val' + '_questions.json'
with open(valid_questions_path) as f:
    valid_data = json.load(f)

valid_data = valid_data['questions']
valid_text,valid_y=y_valid(valid_data,labels)
valid_text = x_text_valid(valid_text,tokenizer)

valid_steps_per_epoch = int(validation_sample/valid_batch_size)

valid_gen = valid_generator(valid_data,validation_sample,vocab_size,sequence_length,valid_text,valid_y,num_labels,valid_batch_size)

#####################################################33

model=relational_model()
model.compile(optimizer=Adam(lr=learning_rate),loss='categorical_crossentropy', metrics=['accuracy'])

####### Load Weights ##########33

#model.load_weights("RN_weights/Batch_size:32_spe:21874epochs:25.0.h5")

print(model.summary())
model_name = "Valid_RN_weights_new/Batch_size:"+str(batch_size)+"_"+"spe:"+str(steps_per_epoch)+"epochs:"+str(epochs)

for ep in range( epochs ):
    print("Starting Epoch" , ep )
    history=model.fit_generator(generator=train_gen,steps_per_epoch=steps_per_epoch,validation_data=valid_gen,validation_steps=valid_steps_per_epoch,epochs=1,verbose=1)
    with open("Valid_RN_weights_new/hist_"+str(ep)+".pickle","wb") as f:
        pickle.dump(history.history,f,protocol=pickle.HIGHEST_PROTOCOL)
    if not model_name is None:
        model.save_weights( model_name + "." + str( ep )+".h5")
    print("Finished Epoch" , ep )
