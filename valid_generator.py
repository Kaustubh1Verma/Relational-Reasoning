from __future__ import print_function
import json
import os.path
import random as ra
import numpy as np
from scipy import ndimage, misc
import itertools
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical


def valid_generator(data,n,vocab_size, sequence_length,x_text,ans_labels,num_labels,batch_size):
    
    path = 'CLEVR_v1.0'
    questions_path = path + '/questions/CLEVR_' + 'val' + '_questions.json'
    images_path = path + '/'+ 'images/val' + '/'

    x_text = x_text[0:n]
    ans_labels = ans_labels[0:n]
    data = data[0:n]
    

    #this line is just to make the generator infinite, keras needs that    
    while True:

        x_text1 = []     # List of questions
        x_image = []    # List of images
        y_list=[]       
        images = {}     # Dictionary of images, to minimize number of imread ops
        
        zipped=itertools.cycle(zip(x_text,ans_labels,[x['image_filename'] for x in data]))
        
        for _ in range( batch_size) :
            ques,ans,name = next(zipped)

            # Create an index for each image
            if not name in images:
                images[name] = misc.imread(images_path + name, mode='RGB')

            x_text1.append(ques)
            x_image.append(images[name])

            # Convert labels to categorical labels
            y_list.append(ans)


        # Convert x_image to np array
        x_image = np.array(x_image)
        x_text1 = np.array(x_text1)
        y = to_categorical(y_list, num_labels)
        y = np.array(y)
        y = np.reshape(y,(y.shape[0],1,y.shape[1]))
            
        # print(x_text1.shape)
        # print(x_image.shape)
        # print(y.shape)

        yield [x_text1,x_image] , y


